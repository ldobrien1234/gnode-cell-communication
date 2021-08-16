
import random as rnd

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torch.autograd.functional import jacobian

import matplotlib.pyplot as plt
import seaborn as sns

from data_classes import NODEData

#lets us use our computer's gpu if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#seed for repeatability
#ensures only deterministic convolution algorithms
torch.backends.cudnn.deterministic = True
#does not look for fasest convolution algorithm
torch.backends.cudnn.benchmark = False

torch.manual_seed(0) #seed for random numbers


with torch.no_grad():

    #get dataset from file
    dataset = NODEData("Cell12_Features.txt", "Cell12_TimeSeries.txt")
    #get the trained neural network from the file
    num_eval = dataset.num_eval    
    nCell = dataset.nCell
    
    #same as neural_ode.py
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
    #same as in neural_ode.py
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
        
    model_state_dict = torch.load("Cell12_Model.pt")

    ode_func = ODEFunc()
    
    #use ODE block to incorporate forward pass
    model = ODEBlock(ode_func, T=10)
        
    model.load_state_dict(model_state_dict)
        
    
    model.eval() #initiate the model for evaluation (e.g. no dropout)
    
    num_examples = dataset.num_examples
    #sample a random index to get an example to compare against the model
    sample = rnd.randint(0, num_examples - 1)
 
    #sampling the features and targets
    #features dimension (nCell, no. features)
    #targets dimension (nCell, num_eval, no.features)
    features, targets = dataset.__getitem__(sample)
    

    
    h_final = model(features.float()) #dimension (num_eval, nCell, no. features)
     
    
num_eval = dataset.num_eval
x_axis = torch.arange(0, num_eval)
#getting the features individually
for cell in range(nCell):
    h_finalG = h_final[1:,cell,0]
    targetG = targets[cell,::,0]
    
    
    h_finalP = h_final[1:,cell,1]
    targetP = targets[cell,::,1]
    
    h_finalX = h_final[1:,cell,2]
    targetX = targets[cell,::,2]
    
    plt.figure()
    plt.title("Cell" + str(cell+1))
    
    plt.plot(x_axis, h_finalG, linestyle='solid', color='red', label='prediction')
    plt.plot(x_axis, targetG, linestyle='solid', color='black',label='ground truth')
    
    plt.plot(x_axis, h_finalP, linestyle='solid', color='red')
    plt.plot(x_axis, targetP, linestyle='solid', color='black')
    
    plt.plot(x_axis, h_finalX, linestyle='solid', color='red')
    plt.plot(x_axis, targetX, linestyle='solid', color='black')
    
    plt.legend()
    
    plt.show()

#We can also show causal dependencies using the jacobian
#we want to find the derivative of the dynamics of each output with respect to each input
total_J = torch.zeros(2,3,2,3)
#the jacobian gives causal dependencies at a given observation
for i in range(num_eval):
    J_model = jacobian(ode_func.net.eval(), targets[::,i,::])
    
    total_J += J_model
    
J_model = torch.abs(total_J)
J_model = J_model.reshape(6,6)

genes = ["G1", "P1", "X1", "G2", "P2", "X2"]
sns.heatmap(J_model, xticklabels=genes, yticklabels=genes, cmap="YlGnBu")




