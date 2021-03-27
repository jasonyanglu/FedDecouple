from torchvision import datasets, transforms
import torch
from models.models import MLP, CNNMnist, CNNCifar, ResNet, MobileNet, Net, NewNet, customResNet, customMobileNet162, \
    customMobileNet138, customMobileNet150
from utility.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar10_noniid, cifar100_noniid, cifar10_noniid_imbalance


# function to load predefined datasets; can make custom dataloader here as well
# also divide the data for all clients by using sampling.py file present in utility folder
def load_dataset(args):
    '''
    Function to load predefined datasets such as CIFAR-10, CIFAR-100 and MNIST via pytorch dataloader
    Declare Custom Dataloaders here if you want to change the dataset
    Also, the function to split training data among all the clients is called from here
    '''

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        test_dataset = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample clients
        if args.iid:
            clients_idx = mnist_iid(args, train_dataset, args.num_clients)
        else:
            clients_idx = mnist_noniid(args, train_dataset, args.num_clients)
        args.num_classes = 10
        return train_dataset, test_dataset, clients_idx
    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=trans_cifar)
        test_dataset = datasets.CIFAR10(args.path_cifar10, train=False, download=True, transform=trans_cifar)
        if args.imbalance != 0:
            train_clients_idx, client_class_idx = cifar10_noniid_imbalance(args, train_dataset)
            test_clients_idx, _ = cifar10_noniid(args, test_dataset, client_class_idx)
        else:
            clients_idx = cifar10_noniid(args, train_dataset, args.num_clients)
        args.num_classes = 10
        return train_dataset, test_dataset, train_clients_idx, test_clients_idx
    elif args.dataset == 'cifar100':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=trans_cifar)
        test_dataset = datasets.CIFAR100(args.path_cifar100, train=False, download=True, transform=trans_cifar)
        if args.iid:
            clients_idx = cifar_iid(args, train_dataset, args.num_clients)
        else:
            clients_idx = cifar100_noniid(args, train_dataset, args.num_clients)
        args.num_classes = 100
        return train_dataset, test_dataset, clients_idx
    else:
        exit('Error: unrecognized dataset')


# function to use the model architectures present in Nets.py file present in models folder

def load_model(args):
    '''

    Function to load the required architecture (model) for federated learning

    '''

    if args.model == 'cnn' and args.dataset == 'cifar':
        model = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        model = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.model == 'MobileNet':
        model = MobileNet(args=args).to(args.device)
    elif args.model == 'resnet':
        model = ResNet.ResNet34(args=args).to(args.device)
    elif args.model == 'resnet50':
        model = ResNet.ResNet50(args=args).to(args.device)
    elif args.model == 'resnet101':
        model = ResNet.ResNet101(args=args).to(args.device)
    elif args.model == 'resnet152':
        model = ResNet.ResNet152(args=args).to(args.device)
    elif args.model == 'NewNet':
        model = NewNet(args=args).to(args.device)
    elif args.model == 'customMobileNet162':
        model = customMobileNet162(args=args).to(args.device)
    elif args.model == 'customMobileNet150':
        model = customMobileNet150(args=args).to(args.device)
    elif args.model == 'customMobileNet138':
        model = customMobileNet138(args=args).to(args.device)
    elif args.model == 'customResNet204':
        model = customResNet.customResNet204(args=args).to(args.device)
    elif args.model == 'customResNet192':
        model = customResNet.customResNet192(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')

    return model
