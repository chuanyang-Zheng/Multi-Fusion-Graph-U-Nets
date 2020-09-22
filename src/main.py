import argparse
import random
import time
import torch
import numpy as np
from network import GNet
from trainer import Trainer
from utils.data_loader import FileLoader
import logging
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=1, help='seed')
    parser.add_argument('-data', default='DD', help='data folder name')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=2, help='epochs')
    parser.add_argument('-batch', type=int, default=8, help='batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.2, help='drop output')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', type=float, default='0.9 0.8 0.7')
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    parser.add_argument('-dis_frequence', type=int, default=100, help='display loss frequence')
    parser.add_argument('-log_name', type=str, default="log_COLLAB", help='log name')
    args, _ = parser.parse_known_args()
    return args


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def app_run(args, G_data, fold_idx,logger):
    G_data.use_fold_data(fold_idx)
    net = GNet(G_data.feat_dim, G_data.num_class, args)
    trainer = Trainer(args, net, G_data,logger)
    max_acc=trainer.train()
    return max_acc

def main():
    args = get_args()
    print(args)

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s  %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    fileHandle = logging.FileHandler("{}/{}.txt".format("log_dir",args.log_name))
    fileHandle.setFormatter(formatter)
    logger.addHandler(fileHandle)


    set_random(args.seed)
    start = time.time()
    G_data = FileLoader(args).load_data()
    print('load data using ------>', time.time()-start)
    max_acc=[]
    if args.fold == 0:
        for fold_idx in range(10):
            print('start training ------> fold', fold_idx+1)
            acc_result = app_run(args, G_data, fold_idx, logger)
            max_acc.append(acc_result)
    else:
        print('start training ------> fold', args.fold)
        acc_result = app_run(args, G_data, args.fold - 1, logger)
        max_acc.append(acc_result)
    max_acc = np.asarray(max_acc)
    logger.info("Final Result")
    for i in range(len(max_acc)):
        logger.info("{} Max Accuracy {}".format(i, max_acc[i]))
    logger.info("Mean {}".format(max_acc.mean()))
    logger.info("Standard Variance {}".format(max_acc.std()))


if __name__ == "__main__":
    main()
