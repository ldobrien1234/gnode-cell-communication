# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:24:22 2021

@author: Liam O'Brien
"""
import time
import math
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.linalg as LA
import torch.nn as nn
from torch.autograd.functional import jacobian

import dgl
from GCNLayers import GCNLayer1, GNODEFunc, ODEBlock


#lets us use our computer's gpu if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#seed for repeatability
#ensures only deterministic convolution algorithms
torch.backends.cudnn.deterministic = True
#does not look for fasest convolution algorithm
torch.backends.cudnn.benchmark = False

torch.manual_seed(0) #seed for random numbers

#read the dataset we made with Julia#######################################

file = open("cell_dataset_2000.txt", "r")
data_string = "[" + file.read() + "]"
file.close()

#modifying the string, so eval() works and finds nested lists
data_string = data_string.replace(" ","") #removes spaces
data_string = data_string.replace(";",",")
data_string = data_string.replace("]\t[","],[")
data_string = data_string.replace("]\n[","],[")

#convert the string to a list of lists
dataset = eval(data_string)

#convert the list to a torch tensor
dataset = torch.tensor(dataset)

#Preparing the data##########################################################

#In our dataset is a 3D tensor
#dim 1 gives each training examples (size # examples)
#dim 2 gives each cell (size # cells)
#dim 3 gives each feature (size # features)

#removing A0 from the features
dataset = dataset[::, ::, :3]

#the number of training examples in the data 
num_examples = int(dataset.size()[0] / 2)

#separating the inputs from the outputs
input_features = dataset[:num_examples, ::, ::]
output_features = dataset[num_examples:, ::, ::]

#creating training and test sets using a 60/40 split
num_train = round((num_examples / 10)*6)


X_train = input_features[:num_train, ::, ::]
X_test = input_features[num_train:, ::, ::]

Y_train = output_features[:num_train, ::, ::]
Y_test = output_features[num_train:, ::, ::]

#Creating the graph structure################################################

#The topology for this dataset is given by the following matrix
# 0 1 1
# 0 0 1
# 0 0 0
#creating three cells indexed 0, 1, and 2 (and we want self-edges)
g_top = dgl.heterograph({ ('cell','interacts','cell'):
                             (torch.tensor([0,1,2, 0,0,1]),
                              torch.tensor([0,1,2, 1,2,2]))})

#Now let's create a normalization tensor (based on the degree of
#each node) to improve learning
degs = g_top.in_degrees().float()
norm = torch.pow(degs, -0.5) #each degree to the power of (-1/2)
norm[torch.isinf(norm)] = 0
# add to dgl.Graph in order for the norm to be accessible at training time
g_top.ndata['norm'] = norm.unsqueeze(1).to(device)




#Now we can define the model################################################
#dynamics defined by two GCN layers
gnn = nn.Sequential(GCNLayer1(g=g_top, in_feats=3, out_feats=64, 
                              dropout=0.5, activation=nn.Softplus()),
                  GCNLayer1(g=g_top, in_feats=64, out_feats=3, 
                            dropout=0.5, activation=None)
                  ).to(device)
                   
gnode_func = GNODEFunc(gnn)


#ODEBlock class let's us use an ode solver to find our input at a later time
gnode = ODEBlock(gnode_func, method = 'implicit_adams', atol=1e-3, rtol=1e-4,
                 adjoint = True) 
model = gnode

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = nn.L1Loss()


train_steps = len(X_train)
test_steps = len(X_test)
minibatch = 10 #size of minibatch
train_n_batches = int(train_steps / minibatch)
test_n_batches = int(test_steps / minibatch)

num_grad_steps = 0

train_MAPE_plt = []
train_y_plt = []
#running the training loop

for batch in range(train_n_batches):
    model.train() #modifies forward for training (e.g. performs dropout)
        
    model.gnode_func.nfe = 0 #zero nfe for next forward pass

    batch_X = X_train[batch*minibatch:(batch+1)*minibatch]
    batch_Y = Y_train[batch*minibatch:(batch+1)*minibatch]
    
    MAPEs = []
    losses = []
    for observation in range(minibatch):
            
        start_time = time.time()
        y_pred = model(batch_X[observation])
        f_time = time.time() - start_time #store time for forward pass
        
        nfe = model.gnode_func.nfe #store nfe in forward pass
            
        y = batch_Y[observation]
        
        MAPE = LA.norm((torch.abs(y - y_pred) / torch.abs(y)), 
                        ord=2).detach().item()
        loss = criterion(y_pred,y)
            
        losses.append(loss)
        MAPEs.append(MAPE)
        
    MAPE_avg = sum(MAPEs)  / len(MAPEs)
    loss_avg = sum(losses) / len(losses) #averaging over the minibatch
    
        
    opt.zero_grad()
        
    start_time = time.time()
    loss_avg.backward()
    b_time = time.time() - start_time #store time for backward pass
    
    #avoid exploding gradient
    nn.utils.clip_grad_norm_(model.parameters(), 1e+5)
        
    opt.step()
    num_grad_steps += 1
    
    train_loss = loss_avg.item()
    
    print("MAPE = ", MAPE_avg)
    print("train loss = ", train_loss)
    if math.isnan(train_loss):
        break
    train_MAPE_plt.append(MAPE_avg)
    train_y_plt.append(train_loss)


#using testing data
with torch.no_grad():
    model.eval() #modifies forward for evaluation (e.g. no dropout)
    
    test_MAPE_plt = []
    test_y_plt = []
    for batch in range(test_n_batches):
        batch_X = X_train[batch*minibatch:(batch+1)*minibatch]
        batch_Y = Y_train[batch*minibatch:(batch+1)*minibatch]
        
        MAPEs = []
        losses = []
        for observation in range(minibatch):
            
            start_time = time.time()
            y_pred = model(batch_X[observation])
            f_time = time.time() - start_time #store time for forward pass
        
            nfe = model.gnode_func.nfe #store nfe in forward pass
            
            y = batch_Y[observation]
        
            MAPE = LA.norm((torch.abs(y - y_pred) / torch.abs(y)),
                            ord=2).item()
            loss = criterion(y_pred,y)
            
            MAPEs.append(MAPE)
            losses.append(loss)
        
        MAPE_avg = sum(MAPEs)  / len(MAPEs)
        loss_avg = sum(losses) / len(losses) #averaging over the minibatch
        
        test_loss = loss_avg.item()
        
        print("MAPE = ", MAPE_avg)
        print("test loss = ", test_loss)
        
        test_MAPE_plt.append(MAPE_avg)
        test_y_plt.append(test_loss)
    
    
     
train_x_plt = np.array(range(len(train_y_plt)))       
train_y_plt = np.array(train_y_plt)

test_x_plt = np.array(range(len(test_y_plt)))
test_y_plt = np.array(test_y_plt)

plt.plot(train_x_plt, train_y_plt, 'blue', label='training loss')
plt.plot(test_x_plt, test_y_plt, 'orange',label='testing loss')
plt.title('Loss')
plt.legend()

       
# making a new plot for MAPE 
plt.figure()

train_MAPE_plt = np.array(train_MAPE_plt)
test_MAPE_plt = np.array(test_MAPE_plt)

plt.plot(train_x_plt, train_MAPE_plt, 'blue', label='training MAPE')
plt.plot(test_x_plt, test_MAPE_plt, 'orange', label='testing MAPE')
plt.title('MAPE')
plt.legend()


#Lastly, we want our model to learn causal dependencies####################

#we trained our model, and now we want to find the derivative of the
#dynamics of each output with respect to each input
total_J = torch.zeros(3,3,3,3)
for observation in range(100):
    #the jacobian gives causal dependencies at a given observation
    J_gnn = jacobian(gnn.eval(), Y_test[observation])
    #summing the jacobian is more revealing of average causal dependencies
    total_J += J_gnn
    
print(J_gnn)
