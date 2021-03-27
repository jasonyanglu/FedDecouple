import copy
import torch
from torch import nn


# since the number of samples in all the users is same, simple averaging works
def FedAvg(w):
    '''

    Function to average the updated weights of client models to update the global model (when the number of samples is same for each client)

    Parameters:

        w (list) : The list of state_dicts of each client

    Returns:

        w_avg (state_dict) : The updated state_dict for global model

    '''

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def customFedAvg(w, weight=1):
    '''

    Function to average the updated weights of client models to update the global model (when the number of samples is same for each client)

    Parameters:

        w (list) : The list of state_dicts of each client

    Returns:

        w_avg (state_dict) : The updated state_dict for global model

    '''

    print(len(w))
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# if number of samples are different, use this function in personal_fed.py
def FedAvgRefined(w, count):
    '''

    Function to average the updated weights of client models to update the global model (clients can have different number of samples)

    Parameters:

        w (list) : The list of state_dicts of each client

        count (list) : The list of number of samples each client has

    Returns:

        w_updated (state_dict) : The updated state_dict for global model after doing the weighted average of local models

    '''

    w_mul = []

    for j in range(len(w)):
        w_avg = copy.deepcopy(w[j])

        for i in w_avg.keys():
            w_avg[i] = torch.mul(w_avg[i], count[0])

        w_mul.append(w_avg)

    w_updated = copy.deepcopy(w_mul[0])

    for k in w_updated.keys():
        for i in range(1, len(w_mul)):
            w_updated[k] += w_mul[i][k]
        w_updated[k] = torch.div(w_updated[k], sum(count))
    return w_updated

