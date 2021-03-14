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

def customFedAvg(w,weight=1):

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
def FedAvgRefined(w,count):

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
            w_avg[i] = torch.mul(w_avg[i],count[0])

        w_mul.append(w_avg)

    w_updated = copy.deepcopy(w_mul[0])

    for k in w_updated.keys():
        for i in range(1, len(w_mul)):
            w_updated[k] += w_mul[i][k]
        w_updated[k] = torch.div(w_updated[k], sum(count))
    return w_updated



# if you want to add random noise for some users with probability p, use this update function
def DiffPrivFedAvg(w):

    '''

    Update global model by incorporating Differential Privacy (Adding noise to the weights of the clients so that their data cannot be reconstructed from model weights)
    
    Current implementation is for same number of data samples per client

    Parameters:

        w (list) : The list of state_dicts of local models

    Returns:

        w_avg (state_dict) : Updated state_dict for global model

    Working:

        p (probability of selecting original weights for a particular client) : Set this value from (0,1) 

        Generate noise:

            Mean : 0

            Standard Deviation : Sum of squares of all the weights divided by total number of elements of the weight tensor

            Shape : Same as that of weight tensor

        Add this generated noise to a copy of weight tensor and use that value for aggregation

    '''


    
    w_new = []
    
    for i in range(len(w)):
        
        a = random.uniform(0,1)
        
        #probability of selecting the original weights
        p = 0.8

        if(a<=p):
            w_new.append(copy.deepcopy(w[i]))
        else:
            w_temp = copy.deepcopy(w[i])
            
            for keys in w_temp.keys():
                
                # copy original model weights

                beta = copy.deepcopy(w_temp[keys])
                
                # convert it to numpy to find sum of squares of its elements

                alpha = w_temp[keys].cpu().numpy()
                
                epsilon = 10**(-8)

                # set very small elements to zero
                
                alpha[np.abs(alpha) < epsilon] = 0
                
                alpha = alpha + 0.000005
                
                ele_square = np.power(alpha,2)
                
                ele_sum = np.sum(ele_square)
                
                # Divide sum of squares value by size of tensor to get standard deviation

                ele_val = ele_sum.item()/alpha.size

                # Generate gaussian noise of same shape as that of model weights

                w_temp[keys] = np.random.normal(0,ele_val,np.shape(w_temp[keys]))
                
                w_temp[keys] = torch.from_numpy(w_temp[keys])
                
                w_temp[keys] = w_temp[keys].type(torch.cuda.FloatTensor)
                
                # Add noise to the weights

                w_temp[keys] = beta + w_temp[keys]
            
            w_new.append(copy.deepcopy(w_temp))
            
    
    w_avg = copy.deepcopy(w_new[0])
    
    for k in w_avg.keys():
        for i in range(1, len(w_new)):
            
            w_avg[k] = w_avg[k].type(torch.cuda.FloatTensor)
            w_new[i][k] = w_new[i][k].type(torch.cuda.FloatTensor)
        
            w_avg[k] += w_new[i][k]
        
        w_avg[k] = torch.div(w_avg[k], len(w_new))
        w_avg[k] = w_avg[k].type(torch.cuda.FloatTensor)
        
    return w_avg   

