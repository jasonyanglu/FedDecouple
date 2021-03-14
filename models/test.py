import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# testing an image

def test_img(net_g, datatest, args):

    '''

    Test Global Model performance on testing data

    Parameters:

        net_g (state_dict) : Global Model

        datatest (dataset) : The testing data

        args (dictionary) : The list of arguments defined by the user

    Returns:

        accuracy (float) : Percentage accuracy on test set of the model

        test_loss (float) : Cumulative loss on the data

    '''

    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    #print (data_loader)
    l = len(data_loader)
        
    with torch.no_grad():
        
        predictions = []
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()
            
            for i in y_pred:
                predictions.append(int(i.item()))

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        
        return accuracy, test_loss