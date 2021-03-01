import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    args = parser.parse_args()
    return args

def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')['state_dict']

    backbone = {}

    for key in checkpoint:
        if 'backbone1' in key:
            backbone[key.replace('backbone1','backbone')] = checkpoint[key]

    torch.save(backbone, out_file)



def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)

if __name__ == '__main__':
    main()