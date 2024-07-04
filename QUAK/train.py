#!/usr/bin/env python

print('Importing all necessary modules...')
import numpy as np
import scipy as sp
import scipy.stats
import itertools
import logging
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as utils
import math
import time
import os
import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal

import torch.nn as nn
import torch.nn.init as init
import sys

sys.path.append("../new_flows")
from flows import RealNVP, Planar, MAF
from models import NormalizingFlowModel
print("...DONE")

print("Checking if GPU is available...")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("...YES!")
else:
    device = torch.device("cpu")
    print("...NO :(")

# Load and process the data

# needs to be adopted for our data stream
def load_normalize_to_tensor(folder_path, feature):
    final_tensors = []
    for file in os.scandir(folder_path):
        data = np.load(file)
        arr = data[feature]
        Y = np.array(arr.tolist())
        # print(Y.shape)
        # print(Y.shape)

        # normalize
        bkg_mean = []
        bkg_std = []

        for i in range(Y.shape[1]):
            mean = np.mean(Y[:,i])
            std = np.std(Y[:,i])
            bkg_mean.append(mean)
            bkg_std.append(std)
            Y[:,i] = (Y[:,i]-mean)/std
        
        final_tensors.append(torch.tensor(Y))
        # print(f"shape of Y is {torch.tensor(Y).shape}")

    our_tensor = torch.vstack(final_tensors) #but this has 3 dimensions, MUST squeeze
    return torch.squeeze(our_tensor)


def dataloader(folder_path, feature, shuffle = True):
    total_PureBkg = load_normalize_to_tensor(folder_path, feature)
    bs = 800
    return utils.DataLoader(total_PureBkg, batch_size=bs, shuffle=shuffle)



# Building the model

####MAF
class VAE_NF(nn.Module):
    def __init__(self, K, D, nfeats):
        super().__init__()
        self.dim = D
        self.K = K
        self.encoder = nn.Sequential(
            nn.Linear(nfeats, 50),
            nn.LeakyReLU(True),
            nn.Linear(50, 30),
            nn.LeakyReLU(True),
            nn.Linear(30, 20),
            nn.LeakyReLU(True),
            nn.Linear(20, D * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(D, 20),
            nn.LeakyReLU(True),
            nn.Linear(20, 30),
            nn.LeakyReLU(True),
            nn.Linear(30, 50),
            nn.LeakyReLU(True),
            nn.Linear(50, nfeats)
        )

        flow_init = MAF(dim=D)
        flows_init = [flow_init for _ in range(K)]
        prior = MultivariateNormal(torch.zeros(D).to(device), torch.eye(D).to(device))
        self.flows = NormalizingFlowModel(prior, flows_init)

    def forward(self, x):
        # Run Encoder and get NF params
        enc = self.encoder(x)
        mu = enc[:, :self.dim]
        log_var = enc[:, self.dim: self.dim * 2]

        # Re-parametrize
        sigma = (log_var * .5).exp()
        z = mu + sigma * torch.randn_like(sigma)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Construct more expressive posterior with NF

        z_k, _, sum_ladj = self.flows(z)

        kl_div = kl_div / x.size(0) - sum_ladj.mean()  # mean over batch

        # Run Decoder
        x_prime = self.decoder(z_k)
        return x_prime, kl_div


# Creating Training Routine

def train(bkgAE_train_iterator, model):
    global n_steps
    train_loss = []
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for batch_idx, x in enumerate(bkgAE_train_iterator):
        start_time = time.time()

        x = x.float().to(device)

        x_tilde, kl_div = model(x)

        mseloss = nn.MSELoss(size_average=False)
        huberloss = nn.SmoothL1Loss(size_average=False)
        #loss_recons = F.binary_cross_entropy(x_tilde, x, size_average=False) / x.size(0)
        loss_recons = mseloss(x_tilde,x ) / x.size(0)
        #loss_recons = huberloss(x_tilde,x ) / x.size(0)
        loss = loss_recons + beta * kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append([loss_recons.item(), kl_div.item()])

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {:5.3f} ms/batch beta:{}'.format(
                batch_idx * len(x), 50000,
                PRINT_INTERVAL * batch_idx / 50000,
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                1000 * (time.time() - start_time),
                beta

            ))

        n_steps += 1



# Creating Evaluation Routine

def evaluate(model, bkgAE_test_iterator, split='valid'):
    global n_steps
    start_time = time.time()
    val_loss = []
    model.eval()

    with torch.no_grad():
        for batch_idx, x in enumerate(bkgAE_test_iterator):

            x = x.float().to(device)

            x_tilde, kl_div = model(x)
            mseloss = nn.MSELoss(size_average=False)
            huberloss = nn.SmoothL1Loss(size_average=False)
            loss_recons = mseloss(x_tilde,x ) / x.size(0)
            loss = loss_recons + beta * kl_div

            val_loss.append(loss.item())

    print('\nEvaluation Completed ({})!\tLoss: {:5.4f} Time: {:5.3f} s'.format(
        split,
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def train_model(data_folder, output_file):
    features = 'jet'
    if features == 'jet':
        nfeats = 7
    
    train_iterator = dataloader(data_folder + '/training', 'jet', shuffle = True)
    validation_iterator = dataloader(data_folder + '/validation', 'jet', shuffle = False)

    model = VAE_NF(nflow, zdim, nfeats).to(device)
    ae_def = {
                "type":"bkg",
                "trainon":"bkg",
                "features":"JetFeats",
                "architecture":"MAF",
                "selection":"monojet",
                "trainloss":"MSELoss",
                "beta":f"%s" %beta,
                "zdimnflow":f"%s %s" %(zdim,nflow),
                "version":f"v{version}"
             }

    BEST_LOSS = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    LAST_SAVED = -1
    PATIENCE_COUNT = 0
    for epoch in range(1, 1000):
        print("Epoch {}:".format(epoch))
        train(train_iterator, model)
        cur_loss = evaluate(model, validation_iterator)
        print(cur_loss)

        if cur_loss <= BEST_LOSS:
            PATIENCE_COUNT = 0
            BEST_LOSS = cur_loss
            LAST_SAVED = epoch
            print("Saving model!")
            torch.save(model.state_dict(),output_file)

        else:
            PATIENCE_COUNT += 1
            print("Not saving model! Last saved: {}".format(LAST_SAVED))
            if PATIENCE_COUNT > 10:
                print(f"############Patience Limit Reached with LR={lr}, Best Loss={BEST_LOSS}")
                break


if __name__ == '__main__':
    # global variables
    version = 0
    zdim = 4
    nflow = 2
    lr = 5e-3
    beta = 0.1 

    N_EPOCHS = 30
    PRINT_INTERVAL = 400
    NUM_WORKERS = 4
    n_steps = 0

    for file in os.scandir('Data'):
        name = file.name
        print(name)
        train_model('Data/' + name, 'Models/' + name + '.h5')
