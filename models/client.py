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

    target_list = dataset.label
    class_count = [i for i in get_class_distribution(dataset).values()]

    # construct reverse sampler
    class_weights_reverse = []
    for count in class_count:
        if count == 0:
            class_weights_reverse.append(0)
        else:
            class_weights_reverse.append(1/count)
        #class_weights_reverse = 1./torch.tensor(class_count, dtype = torch.float)
        #print(class_weights)
    class_weights_reverse = torch.tensor(class_weights_reverse)
    class_weights_all_reverse = class_weights_reverse[target_list]
        #print(class_weights_all)
    Weighted_sampler_reverse = WeightedRandomSampler(weights=class_weights_all_reverse, num_samples=len(class_weights_all_reverse),replacement=True)

    '''
    # construct smooth sampler1
    t = 0.6
    class_temperature = torch.div(class_weights_reverse,t)
        #class_temperature = [x/t for x in class_count]
    class_weights_temp = torch.exp(class_temperature)
    class_weights_sum = torch.sum(class_weights_temp,dtype=float)
    class_weights_smooth1 = torch.div(class_weights_temp,class_weights_sum)
    class_weights_all_smooth1 = class_weights_smooth1[target_list]
    Weighted_sampler_smooth1 = WeightedRandomSampler(weights=class_weights_all_smooth1, num_samples=len(class_weights_all_smooth1),replacement=True)

    # construct smooth sampler2
    t = 0.3
    class_temperature = torch.div(class_weights_reverse,t)
        #class_temperature = [x/t for x in class_count]
    class_weights_temp = torch.exp(class_temperature)
    class_weights_sum = torch.sum(class_weights_temp,dtype=float)
    class_weights_smooth2 = torch.div(class_weights_temp,class_weights_sum)

    class_weights_all_smooth2 = class_weights_smooth2[target_list]
    Weighted_sampler_smooth2 = WeightedRandomSampler(weights=class_weights_all_smooth2, num_samples=len(class_weights_all_smooth2),replacement=True)
    '''

    # construct dataloader
    train_loader1 = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    train_loader2 = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False,sampler=Weighted_sampler_reverse)
    #train_loader_smooth1 = DataLoader(dataset,batch_size=args.train_batch_size,shuffle=False,sampler=Weighted_sampler_smooth1)
    #train_loader_smooth2 = DataLoader(dataset,batch_size=args.train_batch_size,shuffle=False,sampler=Weighted_sampler_smooth2)
    model.train()
    

    # train and update
    optimizer = torch.optim.SGD(model.parameters(), lr=args.finetune_lr)
    
    epoch_loss = []

    for epoch_i in range(args.num_local_finetune_epochs):
        batch_loss = []

        for batch_idx, images_labels in enumerate(zip(train_loader1, train_loader2)):
            images1, labels1 = images_labels[0][0].to(args.device), images_labels[0][1].to(args.device)
            images2, labels2 = images_labels[1][0].to(args.device), images_labels[1][1].to(args.device)
            #images3, lables3 = images_labels[2][0].to(args.device), images_labels[2][1].to(args.device)
            #images4, lables4 = images_labels[3][0].to(args.device), images_labels[3][1].to(args.device)
            
            # update cb
            optimizer.zero_grad() 
            log_probs1 = model(images1,classifier_cb=True)            
            loss1 = loss_func(log_probs1, labels1)   
            loss_cb = loss1
            loss_cb.backward()
            optimizer.step()

            # update rb
            optimizer.zero_grad()
            log_probs2 = model(images2,classifier_rb=True)
            loss2 = loss_func(log_probs2, labels2)
            loss_rb = loss2
            loss_rb.backward()
            optimizer.step()
            '''
            # update sb1
            optimizer.zero_grad()
            log_probs3 = model(images3,classifier_sb1=True)
            loss3 = loss_func(log_probs3,lables3)
            loss_sb1 = loss3
            loss_sb1.backward()
            optimizer.step()

            # update sb2
            optimizer.zero_grad()
            log_probs4 = model(images4,classifier_sb2=True)
            loss4 = loss_func(log_probs4,lables4)
            loss_sb2 = loss4
            loss_sb2.backward()
            optimizer.step()
            '''
            batch_loss.append(loss_cb.item() + loss_rb.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


# function to test a client
def test_client(args, dataset, test_idx, model, cb=False,rb=False,sb1=False,sb2=False):
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
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    #l = len(data_loader.dataset)

    with torch.no_grad():

        for idx, (data, target) in enumerate(data_loader):
            if args.device.type != 'cpu':
                data, target = data.cuda(), target.cuda()
            log_probs1 = model(data,classifier_cb=True)
            log_probs2 = model(data,classifier_rb=True)
            #log_probs3 = model(data,classifier_sb1=True)
            #log_probs4 = model(data,classifier_sb2=True)
            if cb:
                log_probs = log_probs1
            elif rb:
                log_probs = log_probs2
            else:
                log_probs = 0.5*log_probs1+0.5*log_probs2
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

            _, predicted = torch.max(log_probs.data, 1)
            c = (predicted == target.data).squeeze()            
            for i in range(len(target)):
                target_single = target.data[i]
                class_correct[target_single] += c[i]
                class_total[target_single] += 1
        
        # calculate loss for 10 class separately
        '''
        for i in range(10):
            if class_total[i] != 0:
                class_correct[i] = 100 * class_correct[i] / class_total[i]
            else:
                class_correct[i] = 999   
        ''' 

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        return accuracy, test_loss, class_correct

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