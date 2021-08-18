
import math
from typing import Callable

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torchdiffeq

    
class GCNLayer1(nn.Module):
    """
    General GCN Layer
    """
    def __init__(self, g:dgl.heterograph, batch_size:int, nCell:int,
                 in_feats:int, out_feats:int, 
                 activation:Callable[[torch.Tensor], torch.Tensor],
                 dropout:int):
        super().__init__()
        self.g = g
        self.in_feats = in_feats
        self.out_feats = out_feats
        
        
        self.Linear = nn.Linear(in_feats, out_feats)
        self.norm = nn.BatchNorm1d(nCell)
        
        self.batch_size = batch_size
                
        self.activation = activation
        
        self.graphs = [self.g]*self.batch_size #create a list of all graphs
            
        if dropout:
            #randomly zeroes some elements of the input tensor
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
            
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.uniform_(self.Linear.weight)
    
    def forward(self, h):
        #defines the forward pass through the layer given batch tensor h
        #h has dimension (batch_size, nCell, no. features)
        
        if self.dropout:
            h = self.dropout(h) #randomly zeros elements in h
            
        h = self.Linear(h)
        h = self.norm(h)
        
        g_batch = dgl.batch(self.graphs)
        
        features = torch.reshape(h, (3*self.batch_size, self.out_feats))
        g_batch.ndata['h'] = features
        
        g_batch.update_all(fn.copy_src(src = 'h', out = 'm'),
                     fn.sum(msg = 'm', out = 'h'))
        
        h = g_batch.ndata.pop('h')
        h = torch.reshape(h,(self.batch_size, self.g.number_of_nodes(), self.out_feats))

        if self.activation:
            h = self.activation(h)
        
        return h
    
    
    
    
class GNODEFunc(nn.Module):
    """
    General GNODE function class. To be passed to an ODEBlock
    """
    def __init__(self, gnn:nn.Module):
        
        super().__init__()
        self.gnn = gnn
        #setting number of function evaluations (nfe) equal to zero
        self.nfe = 0
        
    def forward(self, t, h): #t and g needed for ode solver
        self.nfe += 1 #counting the number of function evaluations
        h = self.gnn(h)
        return h
    
class ODEBlock(nn.Module):
    """
    ODEBlock defines forward method that uses an ode solver to solve 
    for the final feature matrix.
    """
    def __init__(self, gnode_func:nn.Module, tspan:torch.Tensor=torch.tensor([0,1]),
                 method:str='dopri5', rtol:float=1e-3, atol:float=1e-4, adjoint:bool=True):
        super().__init__()
        self.gnode_func = gnode_func
        self.method = method
        self.adjoint_flag = adjoint
        self.atol, self.rtol = atol, rtol
        self.integration_time = tspan.float()
        
    def forward(self, h:torch.tensor, T:int=1):
        #uses ode solver to solve for output features at time T
        
        self.integration_time = self.integration_time.type_as(h)

        #complete numerical integration
        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(self.gnode_func, h, 
                                             self.integration_time,
                                             rtol=self.rtol, atol=self.atol,
                                             method=self.method)
        else:
            out = torchdiffeq.odeint(self.gnode_func, h, 
                                     self.integration_time,
                                     rtol=self.rtol, atol=self.atol, 
                                     method=self.method)
            
        return out


def compute_MAPE(target:torch.Tensor, output:torch.Tensor):
    elementwise_mape = torch.abs((output - target) / output)
    mape = torch.mean(elementwise_mape)
    return mape
