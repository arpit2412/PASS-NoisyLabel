import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
from train import *

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--noise_mode', default='sym')
    parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
    parser.add_argument('--id', default='')
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--use_pass', action='store_true', help='Use PASS (Peer Agreement based Sample Selection) method')
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)
        cudnn.benchmark = True
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Create checkpoint directory
    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')
    
    # Run training
    if args.use_pass:
        print('Using PASS (Peer Agreement based Sample Selection) method')
    else:
        print('Using standard DivideMix method')
    
    run_dividemix(args)

if __name__ == '__main__':
    main()