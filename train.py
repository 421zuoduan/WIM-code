import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train():
    for e in range(epochs):
        loss_iter = []
        for t, (x, y) in enumerate(loader_train):
            x = x.to(device)
            y = y.to(device)
            model.train()
            scores = model(x)
            loss = F.cross_entropy(scores, y).to(device)
            loss_iter.append(loss.item())

            optimizer_resnet.zero_grad()
            loss.backward()
            optimizer_resnet.step()

            if t%100==0:
                print('Epoch %d, iter %d, loss=%.4f' % (e+1, t,  loss.item()))

        losses.append(np.mean(loss_iter))
        accuracy = evaluate_accuracy(loader_val, model)
        accuracies.append(accuracy)