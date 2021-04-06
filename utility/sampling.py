import numpy as np
from torchvision import datasets, transforms

np.random.seed(0)


def choose_two_digit_imbalance(num_class, num_split):
    split_list = []
    for i in range(num_split):
        while 1:
            x = [i for i in range(num_class)]
            np.random.shuffle(x)
            y = [i for i in range(num_class)]
            np.random.shuffle(y)
            if sum(np.array(x) == np.array(y)) == 0:
                break
        for xx, yy in zip(x, y):
            split_list.append([xx, yy])

    return split_list


# two functions for each type of dataset - one to divide data in iid manner and one in non-iid manner

def mnist_iid(args, dataset, num_clients):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_clients:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_clients)
    dict_clients, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    return dict_clients


def mnist_noniid(args, dataset, num_clients):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_clients:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([], dtype='int64') for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_clients[i] = np.concatenate((dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_clients


def cifar10_noniid(args, dataset, client_class_idx=None):
    num_items = len(dataset)
    dict_clients = {}
    labels = np.arange(10)
    idx_list = [[] for _ in range(10)]

    for idx, i in enumerate(dataset):
        idx_list[i[1]].append(idx)

    # if k = 4, a particular client can have samples only from at max 4 classes
    # k = args.overlapping_classes
    k = 2
    num_examples = int(np.ceil(num_items / args.num_clients / k))

    if client_class_idx is None:
        client_class_idx = choose_two_digit_imbalance(10, k)

    for i in range(args.num_clients):

        min_class = idx_list[client_class_idx[i][0]]
        min_class_idx = set(np.random.choice(min_class, num_examples, replace=False))
        idx_list[client_class_idx[i][0]] = list(set(min_class) - min_class_idx)
        dict_clients[i] = list(min_class_idx)

        maj_class = idx_list[client_class_idx[i][1]]
        maj_class_idx = set(np.random.choice(maj_class, num_examples, replace=False))
        idx_list[client_class_idx[i][1]] = list(set(maj_class) - maj_class_idx)
        dict_clients[i] = np.append(dict_clients[i], list(maj_class_idx))

    return dict_clients, client_class_idx


def cifar10_noniid_imbalance(args, dataset, client_class_idx=None):
    num_items = len(dataset)
    dict_clients = {}
    labels = np.arange(10)
    idx_list = [[] for _ in range(10)]

    for idx, i in enumerate(dataset):
        idx_list[i[1]].append(idx)

    # if k = 4, a particular client can have samples only from at max 4 classes
    # k = args.overlapping_classes
    k = 2
    minority_ratio = 1 / (args.imbalance + 1)
    minority_sample_num = int(np.ceil(num_items / args.num_clients * minority_ratio))
    num_examples = [minority_sample_num, int(num_items / args.num_clients - minority_sample_num)]

    if client_class_idx is None:
        client_class_idx = choose_two_digit_imbalance(10, k)

    for i in range(args.num_clients):

        min_class = idx_list[client_class_idx[i][0]]
        min_class_idx = set(np.random.choice(min_class, num_examples[0], replace=False))
        idx_list[client_class_idx[i][0]] = list(set(min_class) - min_class_idx)
        dict_clients[i] = list(min_class_idx)

        maj_class = idx_list[client_class_idx[i][1]]
        maj_class_idx = set(np.random.choice(maj_class, num_examples[1], replace=False))
        idx_list[client_class_idx[i][1]] = list(set(maj_class) - maj_class_idx)
        dict_clients[i] = np.append(dict_clients[i], list(maj_class_idx))

    return dict_clients, client_class_idx

def cifar10_longtailed(args, dataset, imb_factor):
    num_class = 10
    num_items = len(dataset)
    dict_clients = [[] for i in range(args.num_clients)]
   
    #keyList = [i for i in range(args.num_clients)]
    #for i in keyList:
    #    dict_clients[i] = 0
    #labels = np.arange(10)
    idx_list = [[] for _ in range(10)]
    current_list = []
    classes = [i for i in range(10)]
    #client_class_idx = [classes for _ in range(args.num_clients)]
    for idx, i in enumerate(dataset):
        idx_list[i[1]].append(idx)
    
    img_max = int(np.ceil(num_items / (num_class)))
    img_num_per_class = []
    num_total = 0
    for cls_idx in range(num_class):
        num = img_max * (imb_factor**(cls_idx / (num_class - 1.0)))
        num_total += np.floor(num)
        img_num_per_class.append(int(num))
    
    
    for the_class, the_img_num in zip(classes, img_num_per_class):
        current_class = idx_list[the_class]
        class_idx = set(np.random.choice(current_class, the_img_num, replace=False))
        idx_list[the_class] = list(set(current_class) - class_idx)
        if current_list == []:
            current_list = list(class_idx)
        else:
            current_list = np.append(current_list, list(class_idx))
        
    
    for i in range(args.num_clients):

        current_client = set(np.random.choice(current_list, int(num_total//args.num_clients), replace=False))
        current_list = list(set(current_list) - current_client)
        if dict_clients[i] == []:
            dict_clients[i] = list(current_client)
        else:
            dict_clients[i] = np.append(dict_clients[i], list(current_client))

    
    '''
    img_client = []
    for i in range(args.num_clients):
        img_client.append(img_num_per_class)
        np.random.shuffle(img_num_per_class)
    
    for i in range(args.num_clients):
        for the_class, the_img_num in zip(classes, img_client[i]):
            current_class = idx_list[the_class]
            class_idx = set(np.random.choice(current_class, the_img_num, replace=False))
            idx_list[the_class] = list(set(current_class) - class_idx)
            if dict_clients[i] == []:
                dict_clients[i] = list(class_idx)
            else:
                dict_clients[i] = np.append(dict_clients[i], list(class_idx))
    '''        

    
    return dict_clients, classes

    
    

    

def cifar100_noniid(args, dataset, num_clients):
    num_items = int(len(dataset))
    dict_clients = {}
    labels = [i for i in range(100)]
    idx = {i: np.array([], dtype='int64') for i in range(100)}

    j = 0
    for i in dataset:
        # print(i[1])
        idx[i[1]] = np.append(idx[i[1]], j)
        j += 1
    # print(idx.keys())
    k = args.overlapping_classes

    num_examples = int(num_items / (k * num_clients))
    # print(num_examples)

    for i in range(num_clients):
        # print(i)
        t = 0
        while (t != k):
            j = np.random.randint(0, 99)

            if (len(idx[(i + j) % len(labels)]) >= num_examples):
                rand_set = set(np.random.choice(idx[(i + j) % len(labels)], num_examples, replace=False))
                idx[(i + j) % len(labels)] = list(set(idx[(i + j) % len(labels)]) - rand_set)
                rand_set = list(rand_set)
                if (t == 0):
                    dict_clients[i] = rand_set
                else:
                    dict_clients[i] = np.append(dict_clients[i], rand_set)
                t += 1
    # print(dict_clients[0])
    return dict_clients


def cifar_iid(args, dataset, num_clients):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_clients:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_clients)
    dict_clients, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    return dict_clients


def cifar100_iid(args, dataset, num_clients):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_clients:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_clients)
    dict_clients, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    return dict_clients


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
