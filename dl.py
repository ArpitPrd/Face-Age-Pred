import torch
from torch.utils.data import Dataset, Sampler
import random 
import numpy as np
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms

def norm(data, max_ = 255., min_ = 0.):
    return (data - min_)/max_

class PrepPatches(Dataset):
    def __init__(self, file):
        super(PrepPatches, self).__init__()
        self.df = pd.read_csv(file)
        self.file = self.df['file_path'].tolist()
        self.age = self.df['age'].tolist()
        self.data = zip(self.file, self.age)
        
    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        im = Image.open(self.file[index])
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor() 
            ])
        im = transform(im)
        im = norm(im)
        return im, torch.tensor([self.age[index]])