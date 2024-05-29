import torch
import torch.nn as nn
from tqdm import tqdm
from dl import PrepPatches
import numpy as np

def training(model, loss_fn, opt, tr_dataloader, epoch):
    model.train()
    run_loss = 0.
    counter = 0

    for im, true_age in tqdm(tr_dataloader):
        im = im.cuda()
        true_age = true_age.cuda()
        counter += 1
        opt.zero_grad()
        pred_age = model(im)

        loss = loss_fn(pred_age, true_age)
        loss.backward()
        opt.step()

        run_loss += loss.item()

    loss_cycle = run_loss/counter

    print("Train Loss:", loss_cycle)
    return

def validate(model, loss_fn, val_dataloader):
    model.eval()
    run_loss = 0.
    counter = 0
    
    with torch.no_grad():
        for im, true_age in tqdm(val_dataloader):
            im = im.cuda()
            true_age = true_age.cuda()
            counter += 1
            pred_age = model(im)
            loss = loss_fn(pred_age, true_age)
            run_loss += loss.item()

    loss_cycle = run_loss/counter
    print("val loss:", loss_cycle)
    return loss_cycle
        