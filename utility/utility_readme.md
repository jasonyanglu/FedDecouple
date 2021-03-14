## Modules

### sampling.py

* mnist_iid(), cifar_iid, cifar100_iid() : To distribute the training dataset among the clients in iid manner; randomly selecting data samples for each client; no data sample is repeated 

* mnist_noniid(), cifar_noniid(), cifar100_non_iid() : Making the distribution non-iid by restricting clients to have samples from atmost k classes out of 10/100. Randomly select those k classes after setting its value and give equal number of samples from each selected class to a particular client.

---

### LoadSplit.py

* Load_Dataset()

	* Function to load predefined datasets such as CIFAR-10, CIFAR-100 and MNIST via pytorch dataloader

	* Declare Custom Dataloaders here if you want to change the dataset

    * Also, the function to split training data among all the clients is called from here 

* Load_Model()

	* Function to load the required architecture (model) for federated learning


#### Note:

The CIFAR-10 dataset has 50000 training images and 10000 testing images. The dataset for each client (both test and train) was sampled out of the 50000 images. The 10000 images are not touched anywhere and were used later to test the performance of global model only.