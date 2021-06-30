import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('--in_file', help='input checkpoint filename')
    parser.add_argument('--out_file', help='output checkpoint filename')
    parser.add_argument('--prefix', help='prefix to find')
    args = parser.parse_args()
    return args

def process_checkpoint(in_file, out_file, prefix):
    checkpoint = torch.load(in_file, map_location='cpu')['state_dict']

    backbone = {}

    for key in checkpoint:
        print(key)
        if prefix in key:
            backbone[key.replace(prefix, 'backbone')] = checkpoint[key]

    torch.save(backbone, out_file)



def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file, args.prefix)

if __name__ == '__main__':
    main()