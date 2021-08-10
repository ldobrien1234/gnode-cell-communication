
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

import dgl
from torch.utils.data import DataLoader, random_split
from GCNLayers import GCNLayer1, GNODEFunc, ODEBlock, compute_MAPE
from data_classes import GNODEData


batch_size=1000

#lets us use our computer's gpu if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#seed for repeatability
#ensures only deterministic convolution algorithms
torch.backends.cudnn.deterministic = True
#does not look for fasest convolution algorithm
torch.backends.cudnn.benchmark = False

torch.manual_seed(0) #seed for random numbers

#lets us use our computer's gpu if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#seed for repeatability
#ensures only deterministic convolution algorithms
torch.backends.cudnn.deterministic = True
#does not look for fasest convolution algorithm
torch.backends.cudnn.benchmark = False

torch.manual_seed(0) #seed for random numbers


#get dataset from file
dataset = GNODEData("non_random_dataset.txt")

#split into train and test
size_train = round((dataset.__len__() / 10)*8)
train, test = random_split(dataset, 
                           [size_train, dataset.__len__() - size_train])

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

#Creating the graph structure################################################

#The topology for this dataset is given by the following matrix
# 0 1 1
# 0 0 1
# 0 0 0
#creating three cells indexed 0, 1, and 2 (and we want self-edges)
# g_top = dgl.heterograph({ ('cell','interacts','cell'):
#                               (torch.tensor([0,1,2, 0,0,1]),
#                               torch.tensor([0,1,2, 1,2,2]))})

#complete graph
g_top = dgl.heterograph({ ('cell','interacts','cell'):
                              (torch.tensor([0,1,2, 0,0, 1,1, 2,2]),
                              torch.tensor([0,1,2, 1,2, 0,2, 0,1]))})
    
#Now let's create a normalization tensor (based on the degree of
#each node) to improve learning
degs = g_top.in_degrees().float()
norm = torch.pow(degs, -0.5) #each degree to the power of (-1/2)
norm[torch.isinf(norm)] = 0
# add to dgl.Graph in order for the norm to be accessible at training time
g_top.ndata['norm'] = norm.unsqueeze(1).to(device)




#Now we can define the model################################################
#dynamics defined by two GCN layers
gnn = nn.Sequential(GCNLayer1(g=g_top, in_feats=3, out_feats=40, 
                              dropout=0.5, activation=nn.LeakyReLU()),
                    GCNLayer1(g=g_top, in_feats=40, out_feats=3, 
                            dropout=None, activation=None)
                  ).to(device)
                   
gnode_func = GNODEFunc(gnn)


#ODEBlock class let's us use an ode solver to find our input at a later time
gnode = ODEBlock(gnode_func, method = 'rk4', atol=1e-3, rtol=1e-4,
                 adjoint=False)

model = gnode

opt = torch.optim.Adam(model.parameters(), weight_decay=5e-4)
criterion = nn.MSELoss()


train_epochs = 8
test_epochs = 3 

steps = 0
train_y_plt = []
train_mape_plt = []
#running the training loop
for epoch in range(train_epochs):
    model.train()
    
    for i, (features, targets) in enumerate(train_loader):
            
        features = features.to(device)
        targets = targets.to(device)
        
        outputs = torch.zeros(batch_size, 3, 3)
        for k, feature in enumerate(features):
            
            y_pred = model(feature)
            
            outputs[k, ::, ::]= y_pred
        
        loss = criterion(targets, outputs)
        MAPE = compute_MAPE(targets, outputs).item()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        steps += 1
        
        print("train loss = ", loss.item())
        print("train MAPE = ", MAPE)
        print("steps = ", steps)
        train_y_plt.append(loss.item())
        train_mape_plt.append(MAPE)
        
        


steps = 0
test_y_plt = []
test_mape_plt = []
#running the testing loop
for epoch in range(test_epochs):
    model.eval()
    
    with torch.no_grad():
        
        for i, (features, targets) in enumerate(test_loader):
            
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = torch.zeros(batch_size, 3, 3)
            for k, feature in enumerate(features):
                
                y_pred = model(feature)
                
                outputs[k, ::, ::]= y_pred
            
            loss = criterion(targets, outputs)
            MAPE = compute_MAPE(targets, outputs).item()
            
            steps += 1
            
            print("test loss = ", loss.item())
            print("test MAPE = ", MAPE)
            print("steps = ", steps)
            test_y_plt.append(loss.item())
            test_mape_plt.append(MAPE)
            




train_x_plt = np.array(range(len(train_y_plt)))       
train_y_plt = np.array(train_y_plt)
test_x_plt = np.array(range(len(test_y_plt)))
test_y_plt = np.array(test_y_plt)

plt.title('Loss')
plt.plot(train_x_plt, train_y_plt, 'blue', label='training loss')
plt.plot(test_x_plt, test_y_plt, 'orange', label='test loss')

plt.figure()

train_x_plt = np.array(range(len(train_mape_plt)))
train_mape_plt = np.array(train_mape_plt)
test_x_plt = np.array(range(len(test_mape_plt)))
test_mape_plt = np.array(test_mape_plt)

plt.title('MAPE')
plt.plot(train_x_plt, train_mape_plt, 'blue', label= 'training MAPE')
plt.plot(test_x_plt, test_mape_plt, 'orange', label = 'test MAPE')
    
    
#save the training loss in a file
text_file = open("sample8.txt", "w")
for loss, MAPE in zip(train_y_plt, train_mape_plt):
    message = "loss = " + str(loss) +", MAPE = " + str(MAPE) + "\n"
    text_file.write(message)
text_file.close()

#Lastly, we want our model to learn causal dependencies####################

#we trained our model, and now we want to find the derivative of the
#dynamics of each output with respect to each input
total_J = torch.zeros(3,3,3,3)
for i, (features, targets) in enumerate(test_loader):
    for k, feature in enumerate(features):
        #the jacobian gives causal dependencies at a given observation
        J_gnn = jacobian(gnn.eval(), feature)
        #summing the jacobian is more revealing of average causal dependencies
        total_J += J_gnn
    
print(J_gnn)
