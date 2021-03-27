import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F


class DatasetSplit(Dataset):
    """
    Class DatasetSplit - To get datasamples corresponding to the indices of samples a particular client has from the actual complete dataset
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def train_client(args, dataset, train_idx, model):
    '''
    :param args: The list of arguments defined by the user
    :param dataset: Complete dataset loaded by the Dataloader
    :param train_idx: List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client
    :param model: Client Model
    :return:
    '''

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    train_loader = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.train_batch_size, shuffle=True)
    model.train()

    # train and update
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    epoch_loss = []

    for epoch_i in range(args.num_local_epochs):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = model(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def finetune_client(args, dataset, train_idx, model):
    '''

    Train individual client models

    Parameters:

        model (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    model.train()

    # train and update
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    epoch_loss = []

    for iter in range(1):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = model(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


# function to test a client
def test_client(args, dataset, test_idx, model):
    '''
    :param args: (dictionary) The list of arguments defined by the user
    :param dataset: (dataset) The data on which we want the performance of the model to be evaluated
    :param test_idx: (list) List of indices of those samples from the actual complete dataset that are there in the local dataset of this client
    :param model: (state_dict) Client Model
    :return:
        accuracy: (float) Percentage accuracy on test set of the model
        test_loss: (float) Cumulative loss on the data
    '''

    data_loader = DataLoader(DatasetSplit(dataset, test_idx), batch_size=args.test_batch_size)
    model.eval()
    # print (test_data)
    test_loss = 0
    correct = 0

    l = len(data_loader)

    with torch.no_grad():

        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]

            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        return accuracy, test_loss
