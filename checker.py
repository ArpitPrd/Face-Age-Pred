import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from model import Model
import torchvision.transforms as transforms

def norm(data, max_ = 255., min_ = 0.):
    return (data - min_)/max_


def check(file, mod):
    im = Image.open(file)
    transform = transforms.Compose([
                    transforms.Resize((256, 256)),  
                    transforms.ToTensor()  # Convert the image to a tensor and normalize to [0, 1]
                ])
    im = transform(im)
    im = norm(im)
    im = torch.unsqueeze(im, 0)
    model = Model()
    model.load_state_dict(torch.load(mod))
    model.eval()
    age = model(im)
    print("your age is: ", int(age * (57-13) + 13))


mod = 'checkpoints/reg--75--0.027318043606991672'
file = 'UTKFace/26_1_3_20170109133227729.jpg.chip.jpg'

check(file, mod)