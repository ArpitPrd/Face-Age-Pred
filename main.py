import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from model import Model
from dl import PrepPatches
import numpy as np
import torch.optim as optim
from train import training, validate

def main():
    train_data = PrepPatches('train_data.csv')
    tr_dataloader = DataLoader(train_data, batch_size = 64, shuffle = True, num_workers = 4)

    val_data = PrepPatches('val_data.csv')
    val_dataloader = DataLoader(val_data)

    epochs = 100
    checkpoint_dir = './checkpoints/'
    model_path = checkpoint_dir + '--{}--{}.pth'
    best_model_path, best_loss = '', np.inf

######################################### MODEL #############################################
    
    model = Model()
    model.cuda()

####################################### LOSS Fn #############################################

    loss_fn = nn.MSELoss()
    loss_fn.cuda()

####################################### OPT ##################################################

    opt = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-6, betas = (0.9, 0.99))

###################################### Train ##################################################

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")

        training(model, loss_fn, opt, tr_dataloader, epoch)
        val_loss = validate(model, loss_fn, val_dataloader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = model_path.format(epoch, val_loss)
            torch.save(model.state_dict(), best_model_path)
            print("Best loss:", val_loss)

        if epoch % 25 == 0:
            path = 'checkpoints/reg--{}--{}'.format(epoch, val_loss)
            torch.save(model.state_dict(), path)

if __name__ == '__main__':
    main()