
import matplotlib.pyplot as plt

import statistics
import numpy as np
import torch
import torch.nn as nn

import dgl
from torch.utils.data import DataLoader, random_split
from GCNLayers import GCNLayer2, GNODEFunc, ODEBlock, compute_MAPE
from data_classes import GNODEData

batch_size=100

patience = 50 #for early stopping

train_epochs = 10

#final time of the numerical integrator
t_f = 10

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
dataset = GNODEData("Cell123_Features.txt", "Cell123_TimeSeries.txt")
num_eval = dataset.num_eval
nCell = dataset.nCell


#split into train and test
size_train = round((dataset.__len__() / 10)*5)
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
g_top = dgl.heterograph({('cell','interacts','cell'):
                              (torch.tensor([0,1,2]),
                              torch.tensor([0,1,2]))})


#Now we can define the model################################################
#dynamics defined by two GCN layers
gnn = nn.Sequential(GCNLayer2(g=g_top, batch_size=batch_size, nCell=nCell,
                              in_feats=20, out_feats=64, 
                              dropout=None, activation=nn.LeakyReLU()),
                    GCNLayer2(g=g_top, batch_size=batch_size, nCell=nCell,
                              in_feats=64, out_feats=20, 
                            dropout=None, activation=None)
                  ).to(device)
                   
gnode_func = GNODEFunc(gnn)


#create a tensor of times to evaluate our numerical integrator
eval_int = t_f / num_eval
tspan = torch.empty(num_eval+1)
for i in range(num_eval+1):
    tspan[i] = i*eval_int
    

#ODEBlock class lets us use an ode solver to find our input at a later time
#gnode outputs a tensor of dimension (num_eval, nCell, no. features)
gnode = ODEBlock(gnode_func, tspan=tspan, method = 'rk4', atol=1e-3, 
                 rtol=1e-4, adjoint=True)

#model outputs a tensor of dimension (num_eval, nCell, no. features)
model = nn.Sequential(nn.Linear(3,20),
                      nn.LeakyReLU(),
                      gnode,
                      nn.Linear(20,3),
                      nn.LeakyReLU())
                      

opt = torch.optim.Adam(model.parameters(),lr=1e-2)
criterion = nn.MSELoss()


steps = 0
train_y_plt = []
train_mape_plt = []

test_y_plt = []
test_mape_plt = []
#running the training loop
for epoch in range(train_epochs):

    
    for (train_f, train_t), (test_f, test_t) in zip(train_loader, test_loader):
        model.train()
        
        steps += 1
        
        train_f = train_f.to(device) #dimension (batch_size, nCell, no. features)
        train_t = train_t.to(device) #dimension (batch_size, nCell, num_eval, no. features)
        
        h_final = model(train_f) #dimension (num_eval+1, batch_size, nCell, no. features)

        #make h_final have dimension (batch_size, nCell, num_eval, no. features)
        #now directly comparable with targets
        h_final = h_final[1:,::,::,::].permute(1,2,0,3)
        
        train_loss = criterion(train_t, h_final)
        train_MAPE = compute_MAPE(train_t, h_final)
        
        opt.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1e+5) #avoid exploding gradient
        opt.step()
        
        print("train loss = ", train_loss.item())
        print("train MAPE = ", train_MAPE.item())
        train_y_plt.append(train_loss.item())
        train_mape_plt.append(train_MAPE.item())
    
        #training concurrently with testing
        with torch.no_grad():
            model.eval()
    
                
            test_f = test_f.to(device) #dimension (batch_size, nCell, no. features)
            test_t = test_t.to(device) #dimension (batch_size, nCell, num_eval, no. features)
             
            h_final = model(test_f) #dimension (num_eval+1, nCell, no. features) 
            
            #make h_final have dimension (batch_size, nCell, num_eval, no. features)
            #now directly comparable with targets
            h_final = h_final[1:,::,::,::].permute(1,2,0,3)
            
            test_loss = criterion(test_t, h_final)
            test_MAPE = compute_MAPE(test_t, h_final)
           
            print("test loss = ", test_loss.item())
            print("test MAPE = ", test_MAPE.item())
            print("steps = ", steps)
            test_y_plt.append(test_loss.item())
            test_mape_plt.append(test_MAPE.item())
    
        #if testing loss is increasing, we want to stop training
        #average testing loss over 10 epochs
        if (steps > 2*patience):
            old_test_loss = statistics.mean(test_y_plt[steps-2*patience:steps-patience])
            new_test_loss = statistics.mean(test_y_plt[steps-patience:steps])
            
            if new_test_loss > old_test_loss:
                print("Early stopping"+"\n"+"Training and test loss diverge")
                break
            



#save the trained model in a file
torch.save(model.state_dict(), "GNODE_Model8.pt")



train_x_plt = np.array(range(len(train_y_plt)))       
train_y_plt = np.array(train_y_plt)
test_x_plt = np.array(range(len(test_y_plt)))
test_y_plt = np.array(test_y_plt)

plt.title('Loss')
plt.plot(train_x_plt, train_y_plt, 'blue', label='training loss')
plt.plot(test_x_plt, test_y_plt, 'orange', label='test loss')
plt.legend()

plt.figure()

train_x_plt = np.array(range(len(train_mape_plt)))
train_mape_plt = np.array(train_mape_plt)
test_x_plt = np.array(range(len(test_mape_plt)))
test_mape_plt = np.array(test_mape_plt)

plt.title('MAPE')
plt.plot(train_x_plt, train_mape_plt, 'blue', label= 'training MAPE')
plt.plot(test_x_plt, test_mape_plt, 'orange', label = 'test MAPE')
plt.legend()
    
#save the training loss in a file
text_file = open("gnode_loss8.txt", "w")
for loss, MAPE in zip(train_y_plt, train_mape_plt):
    message = "loss = " + str(loss) +", MAPE = " + str(MAPE) + "\n"
    text_file.write(message)
text_file.close()
