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
from torch.utils.data import Subset, DataLoader, Dataset
from datetime import datetime

from utility.load import load_dataset, load_model
from models.client import train_client, test_client, finetune_client, DatasetSplit, train_client_per,finetune_client_per, test_client_per, MyDataset
from models.fed import FedAvg
import matplotlib.pyplot as plt

torch.manual_seed(7)


def args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--path_cifar10', type=str, default=os.path.join('../data/cifar10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join('../data/cifar100/'))
    parser.add_argument('--imbalance_factor', type=int, default=0.01)
    parser.add_argument('--imbalance', type=int, default=100)

    # fl
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--C', type=float, default=0.4)
    parser.add_argument('--num_rounds', type=int, default=50)
    parser.add_argument('--num_local_epochs', type=int, default=5)
    parser.add_argument('--num_local_finetune_epochs', type=int, default=10)
    parser.add_argument('--finetune', type=bool, default=True)
    parser.add_argument('--base_layers', type=int, default=174)

    # train
    parser.add_argument('--model', type=str, default='decouple')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--finetune_lr', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.9)

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
    num_sample_clients = []
    total = 0
    for i in range(len(train_clients_idx)):
        num_sample_clients.append(len(train_clients_idx[i]))
        total += num_sample_clients[i]


    

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

    

    client_accuracy = [[] for i in range(args.num_clients)]
    client_accuracy2 = [[] for i in range(args.num_clients)]
    # Start training
    logging.info("Training")
    start = time.time()
    total_clients = list(range(args.num_clients))
    acc_train_decouple = []
    loss_train_decouple = []
    acc_test_decouple = []
    loss_test_decouple = []

    acc_test2_decouple = []
    loss_test2_decouple = []
    class_acc_decouple = [[0 for i in range(args.num_rounds)] for _ in range(10)]
    for round_i in range(args.num_rounds):

        print('Round {}'.format(round_i))
        logging.info("---------Round {}---------".format(round_i))
        local_params, local_loss = [], []

        selected_clients = np.random.choice(total_clients, int(args.num_clients * args.C), replace=False)
        print("Selected clients: {}".format(selected_clients))

        # test_acc, test_loss = test_client(args, dataset_test, test_clients_idx[i], local_model[i])

        for client_i in selected_clients:
            
            data_splited = DatasetSplit(dataset_train, train_clients_idx[client_i])
            params, loss = train_client(args, data_splited, model=local_model[client_i])
            local_params.append(params)
            local_loss.append(copy.deepcopy(loss))

            '''
            child_counter = 0
            for child in local_model[i].children():
                print(" child", child_counter, "is:")
                print(child)
                child_counter +=1
            print(" ")
            '''
        # store train and test accuracies before updating local models
        avg_train_loss_decouple = 0
        avg_train_acc_decouple = 0
        for i in selected_clients:
            # logging.info("Client {}:".format(i))
            train_acc, train_loss, class_acc = test_client(args, dataset_train, train_clients_idx[i], local_model[i])
            #test_acc, test_loss = test_client(args, dataset_test, test_clients_idx[i], local_model[i])
            
            
            # logging.info("Training accuracy: {:.3f}".format(train_acc))
            # logging.info("Test accuracy: {:.3f}".format(test_acc))
            # logging.info("")
            # stats[i][round_i]['After Training accuracy'] = train_acc
            # stats[i][round_i]['After Test accuracy'] = test_acc
            # writer.add_scalar(str(i) + '/After Training accuracy', train_acc, round_i)
            # writer.add_scalar(str(i) + '/After Test accuracy', test_acc, round_i)

            avg_train_acc_decouple += train_acc
            avg_train_loss_decouple += train_loss
        avg_train_acc_decouple /= len(selected_clients)
        avg_train_loss_decouple /= len(selected_clients)
        acc_train_decouple.append(avg_train_acc_decouple)
        loss_train_decouple.append(avg_train_loss_decouple)

        logging.info("Average train accuracy: {: .3f}".format(avg_train_acc_decouple))

        stats['After Average'][round_i] = avg_train_acc_decouple
        writer.add_scalar('Average' + '/After Test accuracy', avg_train_acc_decouple, round_i)

        #avg_loss = sum(local_loss) / len(local_loss)
        #logging.info('Average training loss: {:.3f}'.format(avg_loss))

        # update base layers
        # hyperparameter = number of layers we want to keep in the base part
        #base_layers = args.base_layers

        # update global weights
        global_params = FedAvg(local_params,num_sample_clients,selected_clients)

        # copy weight to global_model
        global_model.load_state_dict(global_params)

        


        ###FineTuning
        if args.finetune:
            #print("FineTuning")
            
            personal_params = list(global_params.keys())[216:]
            #personal_params= np.append(personal_params, list(global_params.keys())[10:])
            
            for client_i in selected_clients:
                for i, param in enumerate(local_model[client_i].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad = False
                
                data_splited = DatasetSplit(dataset_train, train_clients_idx[client_i])
                trainloader = DataLoader(data_splited)
                X_train = []
                Y_train = []
                for idx, (images, labels) in enumerate(trainloader):
                    X_train.append((images.squeeze()).tolist())
                    
                    Y_train.append(labels.item())
                
                finetune_dataset = MyDataset(data=torch.FloatTensor(X_train), label=Y_train)
                params, loss = finetune_client(args, finetune_dataset, model=local_model[client_i])
                for i, param in enumerate(local_model[client_i].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad = True

            avg_test_acc_decouple = 0
            avg_test_loss_decouple = 0

            avg_test_acc2_decouple = 0
            avg_test_loss2_decouple = 0

            current_acc = []
            current_acc2 = []
            client_class_acc_decouple = [0 for _ in range(10)]
            client_class_acc_decouple2 = [0 for _ in range(10)]
            #client_class_acc_decouple3 = [0 for _ in range(10)]
            #client_class_acc_decouple4 = [0 for _ in range(10)]
            for i in selected_clients:
                # logging.info("Client {}:".format(i))
                # train_acc, train_loss = test_client(args, dataset_train, train_clients_idx[i], local_model[i])
                test_acc, test_loss, class_acc = test_client(args, dataset_test, test_clients_idx[i], local_model[i])
                test_acc2, test_loss2, class_acc2 = test_client(args, dataset_test, test_clients_idx[i], local_model[i],rb=True)
                # test_acc3, test_loss3, class_acc3 = test_client(args, dataset_test, test_clients_idx[i], local_model[i],sb1=True)
                # test_acc4, test_loss4, class_acc4 = test_client(args, dataset_test, test_clients_idx[i], local_model[i],sb2=True)
                
                # logging.info("Training accuracy after finetune: {:.3f}".format(train_acc))
                # logging.info("Testing accuracy after finetune: {:.3f}".format(test_acc))
                # logging.info("")
                #
                # stats[i][round_i]['After finetune Training accuracy'] = train_acc
                # stats[i][round_i]['After finetune Test accuracy'] = test_acc
                # writer.add_scalar(str(i) + '/After finetune Training accuracy', train_acc, round_i)
                # writer.add_scalar(str(i) + '/After finetune Test accuracy', test_acc, round_i)

                avg_test_acc_decouple += test_acc
                avg_test_loss_decouple += test_loss

                avg_test_acc2_decouple += test_acc2
                avg_test_loss2_decouple += test_loss2

                current_acc.append(test_acc)
                current_acc2.append(test_acc2)

                client_accuracy[i].append(test_acc)
                client_accuracy2[i].append(test_acc2)
                '''
                for i in range(10):
                    client_class_acc_decouple[i] += class_acc[i]
                    client_class_acc_decouple2[i] += class_acc2[i]
                        #client_class_acc_decouple3[i] += class_acc3[i]
                        #client_class_acc_decouple4[i] += class_acc4[i]
                '''
                    
            print("cb+rb:")
            print(current_acc)
            print("rb:")
            print(current_acc2)

            # average accuracy for rb+cb
            avg_test_acc_decouple /= len(selected_clients)
            avg_test_loss_decouple /= len(selected_clients)
            # average accuracy for rb
            avg_test_acc2_decouple /= len(selected_clients)
            avg_test_loss2_decouple /= len(selected_clients)

            logging.info("Finetuned average test accuracy: {: .3f}".format(avg_test_acc_decouple))

            # append each epoch cb+rb
            acc_test_decouple.append(avg_test_acc_decouple)
            loss_test_decouple.append(avg_test_loss_decouple)
            # append each epoch rb
            acc_test2_decouple.append(avg_test_acc2_decouple)
            loss_test2_decouple.append(avg_test_loss2_decouple)
                #stats['After finetune Average'][round_i] = avg_test_acc_decouple

            '''
            for i in range(10):
                client_class_acc_decouple[i] /= len(selected_clients)
                client_class_acc_decouple2[i] /= len(selected_clients)
                #client_class_acc_decouple3[i] /= len(selected_clients)
                #client_class_acc_decouple4[i] /= len(selected_clients)
            
            for i in range(10):
                class_acc_decouple[i][round_i] = client_class_acc_decouple[i]
            '''
        
        # Updating base layers of the clients and keeping the personalized layers same
        for client_i in range(len(selected_clients)):
            
            for param_name in list(global_params.keys())[0:216]:
                local_params[client_i][param_name] = copy.deepcopy(global_params[param_name])
            local_model[selected_clients[client_i]].load_state_dict(local_params[client_i])

    end = time.time()

    logging.info("Training Time: {}s".format(end - start))
    logging.info("End of Training")

    np.savetxt('./save/train_loss_decouple.txt', loss_train_decouple)
    np.savetxt('./save/train_accuracy_decouple.txt', acc_train_decouple)
    np.savetxt('./save/test_loss_cb+rb_decouple.txt', loss_test_decouple)
    np.savetxt('./save/test_accuracy_cb+rb_decouple.txt', acc_test_decouple)
    np.savetxt('./save/test_loss_rb_decouple.txt', loss_test2_decouple)
    np.savetxt('./save/test_accuracy_rb_decouple.txt', acc_test2_decouple)
    for i in range(args.num_clients):
        save_dir = './save/accuracy_cb+rb_client_'
        save_dir += str(i)
        np.savetxt(save_dir,client_accuracy[i])
    for i in range(args.num_clients):
        save_dir = './save/accuracy_rb_client_'
        save_dir += str(i)
        np.savetxt(save_dir,client_accuracy2[i])



    '''
    plt.figure()
    plt.plot(range(len(loss_train_decouple)),loss_train_decouple)
    plt.ylabel('train_loss')
    plt.legend()
    plt.savefig('./save/fed_{}_{}_{}_C{}_train_loss_balance.png'.format(args.dataset, args.model, args.num_rounds, args.C))

    plt.figure()
    plt.plot(range(len(acc_train_decouple)),acc_train_decouple)
    plt.ylabel('train_accuracy')
    plt.legend()
    plt.savefig('./save/fed_{}_{}_{}_C{}_train_accuracy_balance.png'.format(args.dataset, args.model, args.num_rounds, args.C))

        

    plt.figure()
    plt.plot(range(len(loss_test_decouple)),loss_test_decouple)
    plt.ylabel('test_loss')
    plt.legend()
    plt.savefig('./save/fed_{}_{}_{}_C{}_test_loss_balance.png'.format(args.dataset, args.model, args.num_rounds, args.C))

    
    plt.figure()
    plt.plot(range(len(acc_test_decouple)),acc_test_decouple)
    plt.ylabel('test_accuracy')
    plt.legend()
    plt.savefig('./save/fed_{}_{}_{}_C{}_test_accuracy_balance.png'.format(args.dataset, args.model, args.num_rounds, args.C))

    
    

    x = [i for i in range(10)]
    width = 0.4
    plt.figure()
    plt.bar(x,client_class_acc_decouple,width=width,label='cb+rb')
    for i in range(10):
        x[i] = x[i] + width
    plt.bar(x,client_class_acc_decouple2,width=width,label='rb')
    
    plt.legend()
    plt.savefig('./save/fed_{}_{}_{}_C{}_test_accuracy_compare.png'.format(args.dataset, args.model, args.num_rounds, args.C))

    x = [i for i in range(args.num_rounds) ]
    plt.figure()
    for i in range(10):
        plt.plot(x,class_acc_decouple[i], label='line'+str(i))
        print("class"+str(i)+": ")
        print( class_acc_decouple[i])
    
    plt.legend()
    plt.savefig('./save/fed_{}_{}_{}_C{}_class_accuracy_compare.png'.format(args.dataset, args.model, args.num_rounds, args.C))
    
    
    # save model parameters
    #torch.save(global_model.state_dict(), './state_dict/server_{}.pt'.format(file_name))
    #for i in range(args.num_clients):
    #    torch.save(local_model[i].state_dict(), './state_dict/client_{}_{}.pt'.format(i, file_name))

    '''
    
def FedPer():
    # Initialize argument dictionary
    args = args_parser()
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
    acc_test_per = []
    loss_test_per = []
    for round_i in range(args.num_rounds):

        print('Round {}'.format(round_i))
        logging.info("---------Round {}---------".format(round_i))
        local_params, local_loss = [], []

        selected_clients = np.random.choice(total_clients, int(args.num_clients * args.C), replace=False)
        print("Selected clients: {}".format(selected_clients))

        # test_acc, test_loss = test_client(args, dataset_test, test_clients_idx[i], local_model[i])

        for client_i in selected_clients:

            
            params, loss = train_client_per(args,dataset_train,train_clients_idx[client_i], net=local_model[client_i])
            local_params.append(params)
            local_loss.append(copy.deepcopy(loss))

            
        # store train and test accuracies before updating local models
        avg_train_acc1 = 0
        avg_train1 = []
        for i in selected_clients:
            # logging.info("Client {}:".format(i))
            train_acc, train_loss = test_client_per(args, dataset_train, train_clients_idx[i], local_model[i])
            #test_acc, test_loss = test_client_per(args, dataset_test, test_clients_idx[i], local_model[i])
            
            
            # logging.info("Training accuracy: {:.3f}".format(train_acc))
            # logging.info("Test accuracy: {:.3f}".format(test_acc))
            # logging.info("")
            # stats[i][round_i]['After Training accuracy'] = train_acc
            # stats[i][round_i]['After Test accuracy'] = test_acc
            # writer.add_scalar(str(i) + '/After Training accuracy', train_acc, round_i)
            # writer.add_scalar(str(i) + '/After Test accuracy', test_acc, round_i)

            avg_train_acc1 += train_acc
        avg_train_acc1 /= len(selected_clients)
        avg_train1.append(avg_train_acc1)
        #logging.info("Average test accuracy: {: .3f}".format(avg_acc1))
        #stats['After Average'][round_i] = avg_acc1
        #writer.add_scalar('Average' + '/After Test accuracy', avg_acc1, round_i)

        avg_loss = sum(local_loss) / len(local_loss)
        #logging.info('Average training loss: {:.3f}'.format(avg_loss))

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
            #print("FineTuning")
            
            personal_params = list(global_params.keys())[base_layers:]
            #personal_params= np.append(personal_params, list(global_params.keys())[10:])
            
            for client_i in selected_clients:
                for i, param in enumerate(local_model[client_i].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad = False
                
                params, loss = finetune_client_per(args, dataset_train,train_clients_idx[client_i], net=local_model[client_i])
                for i, param in enumerate(local_model[client_i].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad = True

            avg_test_acc2 = 0
            avg_test_loss2 = 0
            current_acc = []
            for i in selected_clients:
                test_acc, test_loss = test_client_per(args, dataset_test, test_clients_idx[i], local_model[i])
                
                avg_test_acc2 += test_acc
                avg_test_loss2 += test_loss
                current_acc.append(test_acc)

            print(current_acc)
            avg_test_acc2 /= len(selected_clients)
            avg_test_loss2 /= len(selected_clients)
            acc_test_per.append(avg_test_acc2)
            loss_test_per.append(avg_test_loss2)
            #logging.info("Finetuned average test accuracy: {: .3f}".format(avg_test_acc2))
            
            #stats['After finetune Average'][round_i] = avg_test_acc2
    plt.figure()
    plt.plot(range(len(loss_test_per)),loss_test_per)
    plt.ylabel('test_loss')

    plt.savefig('./save/fedper_{}_{}_{}_C{}_loss_balance.png'.format(args.dataset, args.model, args.num_rounds, args.C))

    plt.figure()
    #plt.plot(range(len(acc_test_decouple)),acc_test_decouple) 
    plt.plot(range(len(acc_test_per)),acc_test_per)
    plt.ylabel('test_accuracy')

    plt.savefig('./save/fedper_{}_{}_{}_C{}_accuracy_balance.png'.format(args.dataset, args.model, args.num_rounds, args.C))

def fedAvg():
    # Initialize argument dictionary
    args = args_parser()
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
    num_sample_clients = []
    num_sample_test = []
    total = 0
    for i in range(len(train_clients_idx)):
        num_sample_clients.append(len(train_clients_idx[i]))
        num_sample_test.append(len(test_clients_idx[i]))
        total += num_sample_clients[i]

    # Initialize Global Server Model
    global_model = load_model(args=args)
    global_model.train()

    # copy weights
    global_params = global_model.state_dict()

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
    acc_test_avg = []
    loss_test_avg = []
    client_accuracy = [[] for _ in range(args.num_clients)]
    for round_i in range(args.num_rounds):

        print('Round {}'.format(round_i))
        logging.info("---------Round {}---------".format(round_i))
        local_params, local_loss = [], []

        selected_clients = np.random.choice(total_clients, int(args.num_clients * args.C), replace=False)
        print("Selected clients: {}".format(selected_clients))

        # test_acc, test_loss = test_client(args, dataset_test, test_clients_idx[i], local_model[i])

        for client_i in selected_clients:

            
            params, loss = train_client_per(args,dataset_train,train_clients_idx[client_i], net=local_model[client_i])
            local_params.append(params)
            local_loss.append(copy.deepcopy(loss))

            
        # store train and test accuracies before updating local models
        avg_train_acc1 = 0
        avg_train1 = []
        for i in selected_clients:
            # logging.info("Client {}:".format(i))
            train_acc, train_loss = test_client_per(args, dataset_train, train_clients_idx[i], local_model[i])
            #test_acc, test_loss = test_client_per(args, dataset_test, test_clients_idx[i], local_model[i])
            
            
            # logging.info("Training accuracy: {:.3f}".format(train_acc))
            # logging.info("Test accuracy: {:.3f}".format(test_acc))
            # logging.info("")
            # stats[i][round_i]['After Training accuracy'] = train_acc
            # stats[i][round_i]['After Test accuracy'] = test_acc
            # writer.add_scalar(str(i) + '/After Training accuracy', train_acc, round_i)
            # writer.add_scalar(str(i) + '/After Test accuracy', test_acc, round_i)

            avg_train_acc1 += train_acc
        #avg_train_acc1 /= len(selected_clients)
        #avg_train1.append(avg_train_acc1)
        #logging.info("Average test accuracy: {: .3f}".format(avg_acc1))
        #stats['After Average'][round_i] = avg_acc1
        #writer.add_scalar('Average' + '/After Test accuracy', avg_acc1, round_i)

        avg_loss = sum(local_loss) / len(local_loss)
        #logging.info('Average training loss: {:.3f}'.format(avg_loss))

        # update base layers
        # hyperparameter = number of layers we want to keep in the base part
        #base_layers = args.base_layers

        # update global weights
        global_params = FedAvg(local_params,num_sample_clients,selected_clients)

        # copy weight to global_model
        global_model.load_state_dict(global_params)

        # Updating base layers of the clients and keeping the personalized layers same
        for client_i in range(len(selected_clients)):
            
            for param_name in list(global_params.keys()):
                local_params[client_i][param_name] = copy.deepcopy(global_params[param_name])
            local_model[selected_clients[client_i]].load_state_dict(local_params[client_i])
        
        avg_test_acc2 = 0
        avg_test_loss2 = 0
        current_acc = []
        
        for i in selected_clients:
            test_acc, test_loss = test_client_per(args, dataset_test, test_clients_idx[i], local_model[i])
            
            avg_test_acc2 += test_acc
            avg_test_loss2 += test_loss
            current_acc.append(test_acc)

            client_accuracy[i].append(test_acc)

        print(current_acc)
        avg_test_acc2 /= len(selected_clients)
        avg_test_loss2 /= len(selected_clients)
        acc_test_avg.append(avg_test_acc2)
        loss_test_avg.append(avg_test_loss2)
            #logging.info("Finetuned average test accuracy: {: .3f}".format(avg_test_acc2))
            
            #stats['After finetune Average'][round_i] = avg_test_acc2
    
    # save data to txt
    np.savetxt('./save/test_loss_fedavg.txt', loss_test_avg)
    np.savetxt('./save/acc_test_fedavg.txt', acc_test_avg)
    for i in range(args.num_clients):
        save_dir = './save/accuracy_fedavg_client_'
        save_dir += str(i)
        np.savetxt(save_dir,client_accuracy[i])


    plt.figure()
    plt.plot(range(len(loss_test_avg)),loss_test_avg)
    plt.ylabel('test_loss')

    plt.savefig('./save/fedavg_{}_{}_{}_C{}_loss_balance.png'.format(args.dataset, args.model, args.num_rounds, args.C))

    plt.figure()
    #plt.plot(range(len(acc_test_decouple)),acc_test_decouple) 
    plt.plot(range(len(acc_test_avg)),acc_test_avg)
    plt.ylabel('test_accuracy')

    plt.savefig('./save/fedavg_{}_{}_{}_C{}_accuracy_balance.png'.format(args.dataset, args.model, args.num_rounds, args.C))
if __name__ == '__main__':
    main()
    #FedPer()
    #fedAvg()
    '''
    accuracy_cb_rb = []
    accuracy_rb = []
    accuracy_cb_rb = np.genfromtxt('./save/decouple_data/test_accuracy_cb+rb_decouple.txt')
    accuracy_rb = np.genfromtxt('./save/decouple_data/test_accuracy_rb_decouple.txt')
    plt.figure()
    plt.plot(range(len(accuracy_cb_rb)),accuracy_cb_rb,label='cb+rb')
    plt.plot(range(len(accuracy_cb_rb)),accuracy_rb,label='rb')
    plt.ylabel('test_accuracy')
    plt.legend()
    plt.savefig('./save/fedDecouple_test_accuracy.png')

    accuracy_rb_client = []
    plt.figure()
    
    for i in range(20):
        save_dir = './save/decouple_data/accuracy_rb_client_'
        save_dir += str(i)
        accuracy_rb_client=np.genfromtxt(save_dir)
        plt.plot(range(len(accuracy_rb_client)),accuracy_rb_client,label=str(i))
    
    plt.ylabel('client_accuracy')
    plt.legend()
    plt.savefig('./save/fedDecouple_client_accuracy.png')
    '''
    
    
    