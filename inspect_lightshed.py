import argparse
import collections
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_dir', required=True, help='The path to the checkpoint file')
    arg_list = parser.parse_args()

    checkpoint = torch.load(arg_list.pth_dir, map_location='cpu')
    for i in checkpoint.keys():
        print(i)
        if type(checkpoint[i]) is dict or type(checkpoint[i]) is collections.OrderedDict:
            for j in checkpoint[i].keys():
                print(f'    {j}')