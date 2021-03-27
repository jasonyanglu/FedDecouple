import torch
import time
import logging
import json
import copy
import os
import numpy as np
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime

from utility.load import load_dataset, load_model
from models.client import train_client, test_client, finetune_client
from models.fed import FedAvg

torch.manual_seed(0)


def args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--path_cifar10', type=str, default=os.path.join('../data/cifar10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join('../data/cifar100/'))
    parser.add_argument('--imbalance', type=int, default=10)

    # fl
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--C', type=float, default=0.4)
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--num_local_epochs', type=int, default=20)
    parser.add_argument('--num_local_finetune_epochs', type=int, default=5)
    parser.add_argument('--finetune', type=bool, default=True)
    parser.add_argument('--base_layers', type=int, default=216)

    # train
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.1)

    # environment
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    return args


def test_model():
    pass


def main():

    # Initialize argument dictionary
    args = args_parser()
    print(args)

    # Taking hash of config values and using it as filename for storing model parameters and logs
    param_str = str(args)
    file_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save configurations by making a file using hash value
    if not os.path.exists('./config'):
        os.makedirs('./config')
    with open('./config/parser_{}.txt'.format(file_name), 'w') as outfile:
        json.dump(args.__dict__, outfile, indent=4)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    SUMMARY = os.path.join('./results', file_name)
    args.summary = SUMMARY
    if not os.path.exists(SUMMARY):
        os.makedirs(SUMMARY)

    # Setting the device - GPU or CPU
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Load dataset
    dataset_train, dataset_test, train_clients_idx, test_clients_idx = load_dataset(args=args)

    # Initialize Global Server Model
    global_model = load_model(args=args)
    global_model.train()

    # copy weights
    global_params = global_model.state_dict()

    # Set up log file
    if not os.path.exists('./log'):
        os.makedirs('./log')
    logging.basicConfig(filename='./log/{}.log'.format(file_name), format='%(message)s', level=logging.DEBUG)

    tree = lambda: defaultdict(tree)
    stats = tree()
    writer = SummaryWriter(args.summary)

    # exit()
    # local models for each client
    local_model = {}

    for i in range(0, args.num_clients):
        local_model[i] = load_model(args=args)
        local_model[i].train()
        local_model[i].load_state_dict(global_params)

    # Start training
    logging.info("Training")
    start = time.time()
    total_clients = list(range(args.num_clients))

    for round_i in range(args.num_rounds):

        print('Round {}'.format(round_i))
        logging.info("---------Round {}---------".format(round_i))
        local_params, local_loss = [], []

        selected_clients = np.random.choice(total_clients, int(args.num_clients * args.C), replace=False)
        print("Selected clients: {}".format(selected_clients))

        test_acc, test_loss = test_client(args, dataset_test, test_clients_idx[i], local_model[i])

        for client_i in selected_clients:
            params, loss = train_client(args, dataset_train, train_clients_idx[client_i], model=local_model[client_i])
            local_params.append(params)
            local_loss.append(copy.deepcopy(loss))


        # store train and test accuracies before updating local models
        avg_test_acc = 0
        for i in selected_clients:
            # logging.info("Client {}:".format(i))
            # train_acc, train_loss = test_client(args, dataset_train, train_clients_idx[i], local_model[i])
            test_acc, test_loss = test_client(args, dataset_test, test_clients_idx[i], local_model[i])
            # logging.info("Training accuracy: {:.3f}".format(train_acc))
            # logging.info("Test accuracy: {:.3f}".format(test_acc))
            # logging.info("")
            # stats[i][round_i]['After Training accuracy'] = train_acc
            # stats[i][round_i]['After Test accuracy'] = test_acc
            # writer.add_scalar(str(i) + '/After Training accuracy', train_acc, round_i)
            # writer.add_scalar(str(i) + '/After Test accuracy', test_acc, round_i)

            avg_test_acc += test_acc
        avg_test_acc /= len(selected_clients)
        logging.info("Average test accuracy: {: .3f}".format(avg_test_acc))

        stats['After Average'][round_i] = avg_test_acc
        writer.add_scalar('Average' + '/After Test accuracy', avg_test_acc, round_i)

        avg_loss = sum(local_loss) / len(local_loss)
        logging.info('Average training loss: {:.3f}'.format(avg_loss))

        # update base layers
        # hyperparameter = number of layers we want to keep in the base part
        base_layers = args.base_layers

        # update global weights
        global_params = FedAvg(local_params)

        # copy weight to global_model
        global_model.load_state_dict(global_params)

        # Updating base layers of the clients and keeping the personalized layers same
        for client_i in range(len(selected_clients)):
            for param_name in list(global_params.keys())[0:base_layers]:
                local_params[client_i][param_name] = copy.deepcopy(global_params[param_name])
            local_model[selected_clients[client_i]].load_state_dict(local_params[client_i])

        ###FineTuning
        if args.finetune:
            # print("FineTuning")
            personal_params = list(global_params.keys())[base_layers:]
            for client_i in selected_clients:
                for i, param in enumerate(local_model[client_i].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad = False
                params, loss = finetune_client(args, dataset_train, train_clients_idx[client_i], model=local_model[client_i])
                for i, param in enumerate(local_model[client_i].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad = True

            avg_test_acc = 0
            for i in selected_clients:
                # logging.info("Client {}:".format(i))
                # train_acc, train_loss = test_client(args, dataset_train, train_clients_idx[i], local_model[i])
                test_acc, test_loss = test_client(args, dataset_test, test_clients_idx[i], local_model[i])
                # logging.info("Training accuracy after finetune: {:.3f}".format(train_acc))
                # logging.info("Testing accuracy after finetune: {:.3f}".format(test_acc))
                # logging.info("")
                #
                # stats[i][round_i]['After finetune Training accuracy'] = train_acc
                # stats[i][round_i]['After finetune Test accuracy'] = test_acc
                # writer.add_scalar(str(i) + '/After finetune Training accuracy', train_acc, round_i)
                # writer.add_scalar(str(i) + '/After finetune Test accuracy', test_acc, round_i)

                avg_test_acc += test_acc
            avg_test_acc /= len(selected_clients)
            logging.info("Finetuned average test accuracy: {: .3f}".format(avg_test_acc))

            stats['After finetune Average'][round_i] = avg_test_acc

    end = time.time()

    logging.info("Training Time: {}s".format(end - start))
    logging.info("End of Training")

    # save model parameters
    torch.save(global_model.state_dict(), './state_dict/server_{}.pt'.format(file_name))
    for i in range(args.num_clients):
        torch.save(local_model[i].state_dict(), './state_dict/client_{}_{}.pt'.format(i, file_name))


if __name__ == '__main__':
    main()
