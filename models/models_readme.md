## Modules
### Fed.py

#### FedAvg()
#### Summary

* Function to average the updated weights of client models to update the global model (when the number of samples is same for each client

####    Parameters:
* w (list) : The list of state_dicts of each client

####  Returns
* w_avg (state_dict) : The updated state_dict for global model

---

#### FedAvgRefined()
#### Summary
 
*  Function to average the updated weights of client models to update the global model (clients can have different number of samples)

####   Parameters:
 * w (list) : The list of state_dicts of each client
 * count (list) : The list of number of samples each client has

####   Returns:
*  w_updated (state_dict) : The updated state_dict for global model after doing the weighted average of local models

---

#### DiffPrivFedAvg()
####    Summary:
* Update global model by incorporating Differential Privacy (Adding noise to the weights of the clients so that their data cannot be reconstructed from model weights)
    
* Current implementation is for same number of data samples per client

####    Parameters:
*  w (list) : The list of state_dicts of local models

####    Returns:
* w_avg (state_dict) : Updated state_dict for global model

#####    Working:
* Probability of selecting original weights for a particular client (p) : Set this value from (0,1)  (default: 0.9)

* Generate noise:
  * Mean : 0
  * Standard Deviation : Sum of squares of all the weights divided by total number of elements of the weight tensor
 * Shape : Same as that of weight tensor
            
* Add this generated noise to a copy of weight tensor and use that value for aggregation
---

### Nets.py 

* MobileNet: Class for standard MobileNet architecture

* ResNet: Class for standard ResNet architecture
  * Resnet with 18, 34, 50, 101 and 152 layers can be declared. 
  * We used ResNet 34 for our experiments. 
---

### test.py

#### test_img()

#### Summary

* Test Global Model performance on testing data

#### Parameters:

* net_g (state_dict) : Global Model

* datatest (dataset) : The testing data

* args (dictionary) : The list of arguments defined by the user

#### Returns:

* accuracy (float) : Percentage accuracy on test set of the model

* test_loss (float) : Cumulative loss while training
---
### Update.py

#### train_client()
#### Summary
*  Function to train individual client models

#### Parameters:

*  net (state_dict) : Client Model

*  datatest (dataset) : Complete dataset loaded by the Dataloader

*  args (dictionary) : The list of arguments defined by the user

*  train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

#### Returns:
*  net.state_dict() (state_dict) : The updated weights of the client model

* train_loss (float) : Cumulative loss while training
---

#### test_client()
#### Summary
  
* Function to test the performance of the client models on their datasets

#### Parameters:

* net (state_dict) : Client Model

* datatest (dataset) : The data on which we want the performance of the model to be evaluated
* args (dictionary) : The list of arguments defined by the user
*  test_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local dataset of this client

#### Returns:

* accuracy (float) : Percentage accuracy on test set of the model

*  test_loss (float) : Cumulative loss on the data

    