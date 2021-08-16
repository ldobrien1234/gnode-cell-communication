
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
    
