import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torch.utils.data import DataLoader, random_split
from GCNLayers import compute_MAPE
from data_classes import NODEData


#initialize some hyperparameters
batch_size = 100
    
train_epochs = 10
test_epochs = 5


#lets us use our computer's gpu if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#seed for repeatability
#ensures only deterministic convolution algorithms
torch.backends.cudnn.deterministic = True
#does not look for fasest convolution algorithm
torch.backends.cudnn.benchmark = False

torch.manual_seed(0) #seed for random numbers


#get dataset from file using NODEData class in data_classes.py
dataset = NODEData("Cell123_Features.txt", "Cell123_TimeSeries.txt")
#number of evals for time-series data
num_eval = dataset.num_eval

#split into train and test
size_train = round((dataset.__len__() / 10)*8)
train, test = random_split(dataset, 
                           [size_train, dataset.__len__() - size_train])

#use DataLoader class in PyTorch for training and testing
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)


#Define ODEFunc for the forward pass
class ODEFunc(nn.Module):

    def __init__(self):
        super().__init__()
        
        #define neural network architectures
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 3))

        #initialize weight and biases
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    #define forward pass of the neural network
    def forward(self, t, y):
        return self.net(y)

#define ODEBlock class so numerical integration calls the model
class ODEBlock(nn.Module):
    
    def __init__(self, ode_func:nn.Module, method:str='rk4',
                 rtol:float=1e-3, atol:float=1e-4, T:int=1):
        super().__init__()
        self.ode_func = ode_func
        self.method = method
        self.rtol, self.atol = rtol, atol
        self.T = T
    
    #forward pass 
    def forward(self, h:torch.Tensor):
        eval_int = self.T / num_eval
        
        tspan = torch.empty(num_eval+1)
        for i in range(num_eval + 1):
            tspan[i] = i*eval_int
        
        h_final = odeint_adjoint(self.ode_func, h, tspan, method=self.method,
                         rtol=self.rtol, atol=self.atol)
        return h_final
        
#define network architecture
ode_func = ODEFunc()

#use ODE block to incorporate forward pass
model = ODEBlock(ode_func, T=10)

#define optimization method and loss function
opt = torch.optim.Adam(model.parameters())
criterion = nn.L1Loss()



steps = 0 #for counting the number of steps
train_y_plt = [] #saving loss in a list to plot later
train_mape_plt = [] #saving MAPE in a list to plot later
#running the training loop
for epoch in range(train_epochs):
    model.train() #initialize model for training (e.g. use dropout)
    
    #get the minibatches of inputs and targets
    for inputs, targets in train_loader:

        #out the tensors on GPU if possible for faster computations
        inputs = inputs.to(device) #dimension (batch_size, nCell, no. features) 
        targets = targets.to(device) #dimension (batch_size, nCell, num_evals, no. features)
        
        #output matrix from odeint: the features at every evaluation time
        #tensor dimension is (num_evals, batch_size, nCell, no. features)
        h_final = model(inputs.float())

        total_loss = 0
        total_MAPE = 0
        #getting the loss for each evaluation time
        for j in range(num_eval):
            #the features and targets at a certain evaluation step
            y_pred = h_final[(j+1),::,::,::] #the 0th index is t=0
            target = targets[::,::,j,::]  #the 0th index is the 1st eval after 0
            
            #compute the loss and MAPE for one evaluation time
            #divide by num_eval so we don't get nan
            loss = criterion(target, y_pred) / num_eval
            MAPE = compute_MAPE(target, y_pred) / num_eval
            
            #summing all the losses and MAPEs for each evaluation time
            total_loss += loss
            total_MAPE += MAPE.item()

        
        opt.zero_grad() #zero the gradients
        total_loss.backward() #taking the gradients w.r.t. model parameters
        opt.step() #update the model parameters
        
        steps +=1
        
        #print training info and store for plots
        print("train loss = ", total_loss.item())
        print("train MAPE = ", total_MAPE)
        print("steps = ", steps)
        train_y_plt.append(total_loss.item())
        train_mape_plt.append(total_MAPE)
        
        

#the loop below for testing closely follows the above
steps = 0
test_y_plt = []
test_mape_plt = []
#running the testing loop
for epoch in range(test_epochs):
    model.eval() #initializes the model for testing (e.g. no dropout)
    
    with torch.no_grad(): #so we don't take up resources computing gradients
        
        for i, (inputs, targets) in enumerate(test_loader):
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            h_final = model(inputs.float())
            total_loss = 0
            total_MAPE = 0
            for j in range(num_eval):
                y_pred = h_final[(j+1),::,::,::]
                target = targets[::,::,j,::]
                
                loss = criterion(target, y_pred) / num_eval
                MAPE = compute_MAPE(target, y_pred) / num_eval
                
                total_loss += loss
                total_MAPE += MAPE.item()
                
            steps += 1
            
            print("test loss = ", total_loss.item())
            print("test MAPE = ", total_MAPE)
            print("steps = ", steps)
            test_y_plt.append(total_loss.item())
            test_mape_plt.append(total_MAPE)

#save the trained model in a file
torch.save(model.state_dict(), "Cell123_Model.pt")


#Plotting the training and testing loss and MAPE
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
text_file = open("cell123_loss.txt", "w")
for loss, MAPE in zip(train_y_plt, train_mape_plt):
    message = "loss = " + str(loss) +", MAPE = " + str(MAPE) + "\n"
    text_file.write(message)
text_file.close()
