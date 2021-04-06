import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import random
from sklearn import metrics
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F



def get_class_distribution(dataset_obj):
    # idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
    # count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}
    count_dict = {k: 0 for k in range(10)}
    for value in dataset_obj.label:
        #y_lbl = element[1]
        #y_lbl = idx2class[y_lbl]
        
        count_dict[value] += 1
    return count_dict


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    
    def __getitem__(self, idx):
        label = self.label[idx]
        data = self.data[idx]
        return data, label
    
    def __len__(self):
        return len(self.label)


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



def train_client(args, dataset, model):
    '''
    :param args: The list of arguments defined by the user
    :param dataset: Complete dataset loaded by the Dataloader
    :param train_idx: List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client
    :param model: Client Model
    :return:
    '''

    loss_func = nn.CrossEntropyLoss()
    #train_idx = list(train_idx)
    train_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    model.train()

    # train and update
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    epoch_loss = []

    for epoch_i in range(args.num_local_epochs):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(train_loader):

            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = model(images,classifier_cb=True)
            
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def finetune_client(args, dataset, model):
    '''

    :param args: The list of arguments defined by the user
    :param dataset: Complete dataset loaded by the Dataloader
    :param train_idx: List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client
    :param model: Client Model
    :return:
        net.state_dict() (state_dict) : The updated weights of the client model
        train_loss (float) : Cumulative loss while training
    '''
    #criterion = nn.CrossEntropyLoss()
    
    loss_func = nn.CrossEntropyLoss()
    #train_idx = list(train_idx)

    # construct sampler
    target_list = dataset.label
    #target_list = target_list[torch.randperm(len(target_list))]
    class_count = [i for i in get_class_distribution(dataset).values()]
    class_weights = 1./torch.tensor(class_count, dtype = torch.float)
    #print(class_weights)
    class_weights_all = class_weights[target_list]
    #print(class_weights_all)
    Weighted_sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all),replacement=True)

    train_loader1 = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    train_loader2 = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False,sampler=Weighted_sampler)
    model.train()
    

    # train and update
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    
    epoch_loss = []

    for epoch_i in range(args.num_local_finetune_epochs):
        batch_loss = []

        for batch_idx, images_labels in enumerate(zip(train_loader1,train_loader2)):
            images1, labels1 = images_labels[0][0].to(args.device), images_labels[0][1].to(args.device)
            images2, labels2 = images_labels[1][0].to(args.device), images_labels[1][1].to(args.device)
            
            optimizer.zero_grad()
            
            log_probs1 = model(images1,classifier_cb=True)
            log_probs2 = model(images2,classifier_rb=True)
            loss1 = loss_func(log_probs1, labels1)
            loss2 = loss_func(log_probs2, labels2)
            loss = 0.5 * loss1 + 0.5 * loss2
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
            if args.device.type != 'cpu':
                data, target = data.cuda(), target.cuda()
            log_probs1 = model(data,classifier_cb=True)
            log_probs2 = model(data,classifier_rb=True)
            log_probs = 0.5 * log_probs1 + 0.5 * log_probs2
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]

            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        return accuracy, test_loss

def train_client_per(args,dataset,train_idx,net):

    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''


    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset,train_idx), batch_size=args.train_batch_size, shuffle=True)
    net.train()
    
    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []
    
    for iter in range(args.num_local_epochs):   
        batch_loss = []
        
        for batch_idx, (images, labels) in enumerate(ldr_train):
            
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    return net.state_dict(),sum(epoch_loss) / len(epoch_loss)

def finetune_client_per(args,dataset, train_idx, net):

    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''


    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.train_batch_size, shuffle=True)
    net.train()
    
    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []
    
    for iter in range(1):   
        batch_loss = []
        
        for batch_idx, (images, labels) in enumerate(ldr_train):
            
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    return net.state_dict(),sum(epoch_loss) / len(epoch_loss)

def test_client_per(args,dataset,test_idx,net):

    '''

    Test the performance of the client models on their datasets

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : The data on which we want the performance of the model to be evaluated

        args (dictionary) : The list of arguments defined by the user

        test_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local dataset of this client

    Returns:

        accuracy (float) : Percentage accuracy on test set of the model

        test_loss (float) : Cumulative loss on the data

    '''
    
    data_loader = DataLoader(DatasetSplit(dataset, test_idx), batch_size=args.train_batch_size)  
    net.eval()
    #print (test_data)
    test_loss = 0
    correct = 0
    
    l = len(data_loader)
    
    with torch.no_grad():
                
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            
            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        return accuracy, test_loss