
import torch
import numpy as np
from torch.utils.data import Dataset

#lets us use our computer's gpu if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NODEData(Dataset):
    
    def __init__(self, features_file:str, targets_file:str):
        
        f_file = open(features_file, "r")
        self.features = f_file.readlines()
        f_file.close()
        
        t_file = open(targets_file, "r")
        self.targets = t_file.readlines()
        t_file.close()
        
        num_examples = len(self.features)
        
        #selecting any index to get info about dataset
        feature, target = self.__getitem__(index=0)
        
        self.num_eval = target.size()[1]
        self.nCell = target.size()[0]
        self.num_examples = num_examples
        
        
    
    def __getitem__(self, index):
        feature = self.features[index]
        feature = eval(feature)
        feature = torch.tensor(feature)[::,:3]
        
        target = self.targets[index]
        target = target.replace("Any","")
        target = target.replace("Vector{}","")
        target = target.replace("\n","")
        target = eval(target)
        target = torch.tensor(target)
        target = torch.squeeze(target)
        
        return feature, target
    
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