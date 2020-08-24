import argparse
import os
import sys
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter
import json

from data import GenerateData

def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for SRSEG network")
    main_parser.add_argument("--cuda", type=int, required=False, default=0,
                              help="set it to 1 for running on GPU, 0 for CPU")
    main_parser.add_argument("--batch_size", type=int, default=32, help="batch size, default set to 64")
    main_parser.add_argument("--epochs", type=int, default=40, help="epochs, default set to 20")
    main_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    main_parser.add_argument("--n_blocks", type=int, default=4, help="n_blocks, default set to 6")
    main_parser.add_argument("--n_layers", type=int, default=2, help="n_blocks, default set to 6")
    main_parser.add_argument("--n_subs", type=int, default=4, help="n_subs, default set to 8")
    main_parser.add_argument("--n_ovls", type=int, default=0, help="n_ovls, default set to 1")
    main_parser.add_argument("--n_scale", type=int, default=8, help="n_scale, default set to 2")
    main_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")

    main_parser.add_argument("--dataset_name", type=str, default="InriaDataset",
                              help="dataset_name, default set to dataset_name")
    main_parser.add_argument("--img_size", type=int, default=512, help="the img size to crop for training")
    main_parser.add_argument("--up_scale", type=int, default=2, help="the img size to super resolution")

    main_parser.add_argument("--model_title", type=str, default="MIXNet",
                              help="model_title, default set to model_title")
    main_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    main_parser.add_argument("--learning_rate", type=float, default=1.5e-4,
                              help="learning rate, default set to 1e-4")
    main_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    main_parser.add_argument("--save_dir", type=str, default="./trained_model/",
                              help="directory for saving trained models, default is trained_model folder")
    main_parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 7)")

    args = main_parser.parse_args()
    print(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    train(args)



def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    print('===> Loading datasets')
    train_path = './datasets/' + args.dataset_name + '/trains/'
    test_path  = './datasets/' + args.dataset_name + '/tests/'

    train_set = GenerateData(image_dir=train_path, img_size=args.img_size, lr_img_size=args.img_size // args.up_scale, augment=True)
    test_set = GenerateData(image_dir=test_path, img_size=500, lr_img_size=500 // args.up_scale, augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print('===> Building model')


    print("testing")

if __name__ == "__main__":
    main()
