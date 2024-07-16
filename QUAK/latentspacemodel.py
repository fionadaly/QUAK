from models import SimpleNN
from train import VAE_NF
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
# import utils
import torch.utils.data as utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def load_model(filename):
    nflow=2
    zdim=4
    nfeats=7
    model = VAE_NF(nflow, zdim, nfeats).to(device)
    model.load_state_dict(torch.load(filename, map_location=device))
    return model 

def evaluate_for_loss(model, test_iterator, label):
    val_loss = []
    label_array = []
    model.eval()
    with torch.no_grad():
        for x in test_iterator:
            label_array.append(label)

            x = x.float().to(device)

            x_tilde, kl_div = model(x)
            mseloss = nn.MSELoss(size_average=False)
            huberloss = nn.SmoothL1Loss(size_average=False)
            loss_recons = mseloss(x_tilde,x ) / x.size(0)
            loss = loss_recons # + beta * kl_div
            # print(loss)

            val_loss.append(loss.item())
    print(len(val_loss))
    return np.asarray(val_loss), np.array(label_array)


def combine_and_shuffle_data(data_list):
    """
    Combine and shuffle multiple sets of labeled data.
    Args:
        data_list (list): List of tuples (losses tensor, labels tensor)
    Returns:
        tuple: (shuffled_losses tensor, shuffled_labels tensor)
    """
    combined_losses = []
    combined_labels = []

    for losses, labels in data_list:
        combined_losses.append(losses)
        combined_labels.append(labels)

    combined_losses = torch.cat(combined_losses, dim=0)
    combined_labels = torch.cat(combined_labels, dim=0)

    # Shuffle the combined tensor and labels
    indices = torch.randperm(combined_losses.size(0))
    shuffled_losses = combined_losses[indices]
    shuffled_labels = combined_labels[indices]

    # Yield batches
    for start_idx in range(0, len(shuffled_losses), batch_size):
        end_idx = min(start_idx + batch_size, len(shuffled_losses))
        yield shuffled_losses[start_idx:end_idx], shuffled_labels[start_idx:end_idx]

def dataloader(folder, feature = 'jet'):
    tensor_list = []

    for f in os.scandir(folder):
        data = np.load(f)
        arr = data[feature]
        tensor_list.append(arr)
    arr = np.vstack(tensor_list)

    Y = np.array(arr.tolist())

    # normalize
    bkg_mean = []
    bkg_std = []

    for i in range(Y.shape[1]):
        mean = np.mean(Y[:,i])
        std = np.std(Y[:,i])
        bkg_mean.append(mean)
        bkg_std.append(std)
        Y[:,i] = (Y[:,i]-mean)/std

    total_PureBkg = torch.squeeze(torch.tensor(Y))

    bs = 100
    bkgAE_test_iterator = utils.DataLoader(total_PureBkg, batch_size=bs)

    return bkgAE_test_iterator


def save_losses_and_labels(folder_dict, output_file):
    """
    Load event data from npz files, compute losses using VAE models for a specific label,
    standardize the losses, and save the losses and labels to an output npz file.
    Args:
        folder_dict: dictionary mapping folder path to label
        label (int): Label for the events in the npz files.
        output_file (str): Path to the output npz file.
    """
    losses_zjets = []
    losses_qcd = []
    losses_ttbar = []
    labels = []

    for folder, label in folder_dict.items():
        print(folder)
        event_iterator = dataloader(folder)


        loss_qcd, labels_arr = evaluate_for_loss(vae_zjets, event_iterator, label)
        losses_zjets.append(loss_qcd)
        losses_qcd.append(evaluate_for_loss(vae_qcd, event_iterator, label)[0])
        losses_ttbar.append(evaluate_for_loss(vae_ttbar, event_iterator, label)[0])
        labels.append(labels_arr)

    # Combine the losses into one array
    losses_zjets = np.concatenate(losses_zjets)
    losses_qcd = np.concatenate(losses_qcd)
    losses_ttbar = np.concatenate(losses_ttbar)
    labels = np.concatenate(labels)

    combined_losses = np.vstack((losses_zjets, losses_qcd, losses_ttbar))

        #reshaping data
    combined_losses = np.transpose(combined_losses)
    labels = np.transpose(labels)

    # Standardize the losses
    scaler = StandardScaler()
    standardized_losses = scaler.fit_transform(combined_losses)

    # Shuffle the losses and labels
    print(standardized_losses.shape)
    indices = np.arange(standardized_losses.shape[0])
    np.random.shuffle(indices)
    standardized_losses = standardized_losses[indices]
    labels = labels[indices]

    # Save the standardized losses and labels to an npz file
    np.savez(output_file, losses=standardized_losses, labels=labels)

#training model

def train(bkgAE_train_iterator, model):
    global n_steps
    train_loss = []
    model.train()

    for batch_idx, x in enumerate(bkgAE_train_iterator):
        start_time = time.time()

        x = x.float().to(device)

        x_recon = model(x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append([loss_recons.item()])

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {:5.3f} ms/batch beta:{}'.format(
                batch_idx * len(x), 50000,
                PRINT_INTERVAL * batch_idx / 50000,
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                1000 * (time.time() - start_time),
                beta

            ))

        n_steps += 1

def evaluate_DNN(model, test_iterator, split='valid'):
    global n_steps
    start_time = time.time()
    val_loss = []
    model.eval()

    with torch.no_grad():
        for batch_idx, x in enumerate(test_iterator):

            x = x.float().to(device)
            x_recon = model(x)
            loss = model.reconstruction_loss(x, x_recon)

            val_loss.append(loss)

    print('\nEvaluation Completed ({})!\tMean Loss: {:5.4f} Time: {:5.3f} s'.format(
        split,
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)

def train_model(data_folder, output_file):
    lr = 0.001
    model = SimpleNN()

    BEST_LOSS = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    LAST_SAVED = -1
    PATIENCE_COUNT = 0
    for epoch in range(1, 1000):
        print("Epoch {}:".format(epoch))
        train(train_iterator, model)
        cur_loss = evaluate_DNN(model, validation_iterator)
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
    N_EPOCHS = 30
    PRINT_INTERVAL = 400
    NUM_WORKERS = 4
    n_steps = 0

    # loading models
    vae_qcd = load_model('Models/QCD_HT.h5')
    vae_ttbar = load_model('Models/TTTo.h5')
    vae_zjets = load_model('Models/ZJetsToQQ_HT.h5')
    print('models loaded')

    # # creating training arrays with loss
    # train_qcd = load_npz_files('/n/home06/fdaly/QUAK/QUAK/Data/QCD_HT/training', label=0)
    # train_zjets = load_npz_files('/n/home06/fdaly/QUAK/QUAK/Data/ZJetsToQQ_HT/training', label=1)
    # train_ttbar = load_npz_files('/n/home06/fdaly/QUAK/QUAK/Data/TTTo/training', label=2)

    # x_train, y_train = combine_and_shuffle_data([train_qcd, train_zjets, train_ttbar])
    print('\nNow loading training data ...')
    training_folder_dict = {'/n/home06/fdaly/QUAK/QUAK/Data/QCD_HT/training':0,
    '/n/home06/fdaly/QUAK/QUAK/Data/ZJetsToQQ_HT/training':1,
    '/n/home06/fdaly/QUAK/QUAK/Data/TTTo/training':2 }
    save_losses_and_labels(training_folder_dict, '/n/home06/fdaly/QUAK/QUAK/Data/Combined_loss/training.npz')

    print('\nDone loading training data')

    # creating testing arrays with loss
    # val_qcd = load_npz_files('/n/home06/fdaly/QUAK/QUAK/Data/QCD_HT/validation', label=0)
    # val_zjets = load_npz_files('/n/home06/fdaly/QUAK/QUAK/Data/ZJetsToQQ_HT/validation', label=1)
    # val_ttbar = load_npz_files('/n/home06/fdaly/QUAK/QUAK/Data/TTTo/validation', label=2)

    # x_val, y_val = combine_and_shuffle_data([val_qcd, val_zjets, val_ttbar])
    print('\nNow loading validation data ...')
    validation_folder_dict = {'/n/home06/fdaly/QUAK/QUAK/Data/QCD_HT/validation':0,
    '/n/home06/fdaly/QUAK/QUAK/Data/ZJetsToQQ_HT/validation':1,
    '/n/home06/fdaly/QUAK/QUAK/Data/TTTo/validation':2 }

    save_losses_and_labels(validation_folder_dict, '/n/home06/fdaly/QUAK/QUAK/Data/Combined_loss/validation.npz')
    print('\nDone loading validation data')








