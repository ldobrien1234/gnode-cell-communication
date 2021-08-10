import torch
from torch.utils.data import Dataset, DataLoader


class NODEData(Dataset):
    
    def __init__(self, filename:str):
        
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
        #input is a 9-vector
        dataset = torch.reshape(dataset, (dataset.size()[0],1,9))
        
        #the number of training examples in the data 
        self.num_examples = int(dataset.size()[0] / 2)
        
        self.X = dataset[:self.num_examples, ::, ::]
        self.Y = dataset[self.num_examples:, ::, ::]
    
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
