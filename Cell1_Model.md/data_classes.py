
import torch
import numpy as np
from torch.utils.data import Dataset


class NODEData(Dataset):
    
    def __init__(self, features_file:str, targets_file:str):
        
        #since our file has rows and columns, we can use the function below
        features = np.loadtxt(features_file, dtype=np.float64)
        features = torch.from_numpy(features) #convert to torch.Tensor
        features = features[::, :3] #remove the A0 feature (note 2 dimensions)

        num_examples = features.size()[0]

        #open the file containing targets
        file = open(targets_file, "r")
        targets = torch.empty(0)
        #read each line in the file, and convert it to a tensor
        for line in file:
            line = "[" + line.replace("Any", "") + "]"
            line = eval(line)
            line = torch.tensor(line, dtype=torch.float64)
            targets = torch.cat((targets, line))
        file.close()
        
        self.num_eval = targets.size()[1]
        self.num_examples = num_examples
        self.X = features
        self.Y = targets
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.num_examples
    


class GNODEData(Dataset):
    
    def __init__(self, filename:str):
        
        super().__init__()
        
        file = open(filename, "r")
        data_string = "[" + file.read() + "]"
        file.close()
        
        #modifying the string, so eval() works and finds nested lists
        data_string = data_string.replace(" ","") #removes spaces
        data_string = data_string.replace(";",",")
        data_string = data_string.replace("]\t[","],[")
        data_string = data_string.replace("]\n[","],[")
        
        #convert the string to a list of lists
        dataset = eval(data_string)
        
        dataset = torch.tensor(dataset)
        
        #removing A0 from the features
        dataset = dataset[::, ::, :3]
     
        #the number of training examples in the data 
        self.num_examples = int(dataset.size()[0] / 2)
        
        self.X = dataset[:self.num_examples, ::, ::]
        self.Y = dataset[self.num_examples:, ::, ::]
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.num_examples