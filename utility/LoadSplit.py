from torchvision import datasets, transforms
import torch
from models.Nets import MLP, CNNMnist, CNNCifar, ResNet, MobileNet,Net, NewNet, customResNet, customMobileNet162, customMobileNet138, customMobileNet150
from utility.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, cifar100_noniid


# function to load predefined datasets; can make custom dataloader here as well
# also divide the data for all users by using sampling.py file present in utility folder
def Load_Dataset(args):

    ''' 
    Function to load predefined datasets such as CIFAR-10, CIFAR-100 and MNIST via pytorch dataloader

    Declare Custom Dataloaders here if you want to change the dataset

    Also, the function to split training data among all the clients is called from here 
    
    '''
    
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(args,dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(args,dataset_train, args.num_users)
        return dataset_train, dataset_test, dict_users
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(args,dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(args,dataset_train, args.num_users)
        return dataset_train, dataset_test, dict_users
    elif args.dataset =='cifar100':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(args,dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(args,dataset_train, args.num_users)
        return dataset_train, dataset_test, dict_users
    else:
        exit('Error: unrecognized dataset')


# function to use the model architectures present in Nets.py file present in models folder

def Load_Model(args):
    
    '''

    Function to load the required architecture (model) for federated learning

    '''

    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.model == 'MobileNet':
        net_glob = MobileNet(args=args).to(args.device)
    elif args.model == 'ResNet':
        net_glob = ResNet.ResNet34(args=args).to(args.device)
    elif args.model == 'ResNet50':
        net_glob = ResNet.ResNet50(args=args).to(args.device)
    elif args.model == 'ResNet101':
        print('ResNet101')
        net_glob = ResNet.ResNet101(args=args).to(args.device)
    elif args.model == 'ResNet152':
        print('ResNet152')
        net_glob = ResNet.ResNet152(args=args).to(args.device)
    elif args.model =='NewNet':
        net_glob = NewNet(args=args).to(args.device)
    elif args.model =='customMobileNet162':
        net_glob = customMobileNet162(args=args).to(args.device)
    elif args.model =='customMobileNet150':
        net_glob = customMobileNet150(args=args).to(args.device)
    elif args.model =='customMobileNet138':
        net_glob = customMobileNet138(args=args).to(args.device)
    elif args.model =='customResNet204':
        net_glob = customResNet.customResNet204(args=args).to(args.device)
    elif args.model =='customResNet192':
        net_glob = customResNet.customResNet192(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    return net_glob
        
