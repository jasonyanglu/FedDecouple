import numpy as np
from torchvision import datasets, transforms
import random


# two functions for each type of dataset - one to divide data in iid manner and one in non-iid manner

def mnist_iid(args, dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(args, dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_noniid(args, dataset, num_users):
    num_items = int(len(dataset))
    dict_users = {}
    labels = [i for i in range(10)]
    idx = {i: np.array([], dtype='int64') for i in range(10)}

    j = 0
    # print((dataset[0][0]))
    for i in dataset:
        # print(i)
        idx[i[1]] = np.append(idx[i[1]], j)
        j += 1

    # if(num_users<=5):
    #     k = int(10/num_users)
    #     for i in range(num_users):
    #         a = 0
    #         for j in range(i*k,(i+1)*k):
    #             a += j
    #             if(j==i*k):
    #                 dict_users[i] = list(idx[j])
    #             else:
    #                 dict_users[i] = np.append(dict_users[i],idx[j])
    #         print(a)
    #     return dict_users

    # if k = 4, a particular user can have samples only from at max 4 classes
    k = args.overlapping_classes
    # print(idx)
    num_examples = int(num_items / (k * num_users))

    for i in range(num_users):
        t = 0
        while (t != k):
            j = random.randint(0, 9)
            selected_class = (i + j) % len(labels)
            if (len(idx[selected_class]) >= num_examples):
                rand_set = set(np.random.choice(idx[selected_class], num_examples, replace=False))
                idx[selected_class] = list(set(idx[selected_class]) - rand_set)
                rand_set = list(rand_set)
                if (t == 0):
                    dict_users[i] = rand_set
                else:
                    dict_users[i] = np.append(dict_users[i], rand_set)
                t += 1
    return dict_users


def cifar100_noniid(args, dataset, num_users):
    num_items = int(len(dataset))
    dict_users = {}
    labels = [i for i in range(100)]
    idx = {i: np.array([], dtype='int64') for i in range(100)}

    j = 0
    for i in dataset:
        # print(i[1])
        idx[i[1]] = np.append(idx[i[1]], j)
        j += 1
    # print(idx.keys())
    k = args.overlapping_classes

    num_examples = int(num_items / (k * num_users))
    # print(num_examples)

    for i in range(num_users):
        # print(i)
        t = 0
        while (t != k):
            j = random.randint(0, 99)

            if (len(idx[(i + j) % len(labels)]) >= num_examples):
                rand_set = set(np.random.choice(idx[(i + j) % len(labels)], num_examples, replace=False))
                idx[(i + j) % len(labels)] = list(set(idx[(i + j) % len(labels)]) - rand_set)
                rand_set = list(rand_set)
                if (t == 0):
                    dict_users[i] = rand_set
                else:
                    dict_users[i] = np.append(dict_users[i], rand_set)
                t += 1
    # print(dict_users[0])
    return dict_users


def cifar_iid(args, dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar100_iid(args, dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
