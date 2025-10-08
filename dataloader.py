import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

import os, sys

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torchvision.transforms import v2

def get_data_loaders(basePath, input_size:int = 224, batch_size: int = 5):
    transfTrain = [v2.ColorJitter(brightness=.5, hue=.3),
                   v2.RandomHorizontalFlip(p=0.5)]
    
    basePathTrain = os.path.join(basePath, "train")
    trainDataset = FruitDataset(basePathTrain, size=(input_size,input_size), transform=transfTrain)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    basePathVal = os.path.join(basePath, "valid")
    validDataset = FruitDataset(basePathVal, size=(input_size,input_size))
    validLoader = DataLoader(validDataset, batch_size=batch_size, shuffle=True)

    return trainLoader, validLoader

def get_data_loader_eval(basePath, mode:str = "valid", input_size:int = 224, batch_size: int = 5):
    if not mode in ["valid", "test"]:
        raise Exception("Invalid mode")
    
    basePathVal = os.path.join(basePath, mode)
    evalDataset = FruitDatasetEval(basePathVal, size=(input_size,input_size))
    evalLoader = DataLoader(evalDataset, batch_size=batch_size, shuffle=False)

    return evalLoader
    

class FruitDataset(Dataset):
    def __init__(self, path, size=(256, 256), transform:list = None):
        super().__init__()
        self.class2Label = {'banana': 0, 'strawberry': 1, 'grape': 2, 'apple': 3, 'mango': 4}
        self.basePath = path
        self.allow_ext = ('jpg','bmp','png','jpeg','tiff')
        self.size = size
        self.fileNames = [os.path.join(root,file) for root,dirs,files in os.walk(self.basePath) for file in files if file.endswith(self.allow_ext)]
        self.labels = [self.class2Label[os.path.basename(os.path.dirname(file)).lower()] for file in self.fileNames]
        
        self.transform = []
        self.transform.extend([v2.Lambda(lambda x: torch.from_numpy(x).permute(2, 0, 1)),
                               v2.Resize(self.size, antialias=True)])
        if transform:
            self.transform.extend(transform)
        
        self.transform.extend([v2.ToDtype(torch.float32, scale=True),
                               v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.transform = v2.Compose(self.transform)

    def __len__(self,):
        return len(self.fileNames)
    
    def __getitem__(self, idx):        
        imgPath = self.fileNames[idx]
        label = self.labels[idx]
        
        imgTensor = cv2.imread(imgPath)
        imgTensor = cv2.cvtColor(imgTensor,cv2.COLOR_BGR2RGB)
        
        imgTensor = self.transform(imgTensor)

        return imgTensor, label
    
class FruitDatasetEval(FruitDataset):
    def __init__(self, path, size=(224,224)):
        super().__init__(path=path, size=size, transform=None)
    
    def __getitem__(self, idx):
        imgPath = self.fileNames[idx]
        label = self.labels[idx]
        
        imgTensor = cv2.imread(imgPath)
        imgTensor = cv2.cvtColor(imgTensor,cv2.COLOR_BGR2RGB)
        
        imgTensor = self.transform(imgTensor)

        return imgTensor, label, imgPath