# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 15:10:09 2021

@author: Liam O'Brien
"""
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
    def __init__(self, g:dgl.heterograph, in_feats:int, out_feats:int, 
                 activation:Callable[[torch.Tensor], torch.Tensor],
                 dropout:int, bias:bool=True):
        super().__init__()
        self.g = g
        
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
                
        self.activation = activation
            
        if dropout:
            #randomly zeroes some elements of the input tensor
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
            
        self.reset_parameters()
        
    def reset_parameters(self):
        #randomly chooses parameters from uniform distributions
        stdv = 1. / math.sqrt(self.weight.size(1)) #1/sqrt(out_feats)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, h):
        #defines the forward pass through the layer given feature matrix h

        if self.dropout:
            h = self.dropout(h) #randomly zeros elements in h
        g = self.g
        
        h = torch.mm(h, self.weight) #matrix multiply h with weights
        h = h*g.ndata['norm'] #multiply by degree normalization matrix
        
        
        g.ndata['h'] = h #make h the feature matrix of g
        g.update_all(fn.copy_src(src = 'h', out = 'm'),
                     fn.sum(msg = 'm', out = 'h'))
        
        h = g.ndata.pop('h')
        h = h*g.ndata['norm'] #multiply by degree normalization matrix again
        
        if self.bias is not None:
            h = h + self.bias
            
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
    def __init__(self, gnode_func:nn.Module, method:str='dopri5', 
                 rtol:float=1e-3, atol:float=1e-4, adjoint:bool=True):
        super().__init__()
        self.gnode_func = gnode_func
        self.method = method
        self.adjoint_flag = adjoint
        self.atol, self.rtol = atol, rtol
        
    def forward(self, h:torch.tensor, T:int=1):
        #uses ode solver to solve for output features at time T
        
        self.integration_time = torch.tensor([0, T]).float()
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
            
        return out[-1]
