
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
