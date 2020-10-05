import torch
import os, argparse

from data_loader import get_loader
from solver import Solver

"""parsing and configuration"""
def parse_args():
    desc = "ECCV 2018: Deep Recursive HDR"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_name', type=str, default='HDRGAN', help='The type of model')
    parser.add_argument('--data_dir', type=str, default='../Data')
    parser.add_argument('--train_dataset', type=str, default='/database/hdr/HDRv45960/' , help='Train set path')
    parser.add_argument('--test_dataset', type=str, default='/database/hdr/HDRv45960/', help='Test dataset')
    parser.add_argument('--patch_size', type=int, default=256, help='input patch size')
    parser.add_argument('--num_channels', type=int, default=3, help='The number of channels to super-resolve')

    parser.add_argument('--num_threads', type=int, default=24, help='number of threads for data loader to use')
    parser.add_argument('--exposure_value', type=int, default=1, help='exposure value')

    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--save_epochs', type=int, default=1, help='Save trained model every this epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

    parser.add_argument('--save_dir', type=str, default='Result', help='Directory name to save the results')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--gpu_mode', type=bool, default=True)

    parser.add_argument('--stride', type=int, default=32)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --epoch
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --stride
    try:
        assert args.stride < args.patch_size
    except:
        print('it is possible to fail image reconstruction')

    return args

"""main"""
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"    

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode=False")

    # model
    net = Solver(args)

    # train
    net.train()

if __name__ == '__main__':
    main()
