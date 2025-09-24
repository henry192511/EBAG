import argparse
import random
import numpy as np
import utils
import warnings
import torch

from pathlib import Path

warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def get_args():
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = 'cub_norgaprompt'

    subparser = parser.add_subparsers(dest='subparser_name')
    if config == 'cifar100_norgaprompt':
        from configs.cifar100 import get_args_parser
        parser = subparser.add_parser('cifar100_norgaprompt', help='Split-CIFAR100 EBAG configs')
    elif config == "cub_norgaprompt":
        from configs.cub import get_args_parser
        parser = subparser.add_parser('cub_norgaprompt', help='Split-CUB EBAG configs')
    elif config == 'imr_norgaprompt':
        from configs.imr import get_args_parser
        parser = subparser.add_parser('imr_norgaprompt', help='Split-IMR EBAG configs')
    get_args_parser(parser)
    args = parser.parse_args()
    args.config = config
    return args


def get_args():
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_norgaprompt':
        from configs.cifar100 import get_args_parser
        config_parser = subparser.add_parser('cifar100_norgaprompt', help='Split-CIFAR100 EBAG configs')
    elif config == 'imr_norgaprompt':
        from configs.imr import get_args_parser
        config_parser = subparser.add_parser('imr_norgaprompt', help='Split-ImageNet-R EBAG configs')
    elif config == "cub_norgaprompt":
        from configs.cub import get_args_parser
        config_parser = subparser.add_parser('cub_norgaprompt', help='Split-CUB EBAG configs')
    else:
        raise NotImplementedError

    get_args_parser(config_parser)
    args = parser.parse_args()
    args.config = config
    return args


def main(args):

    utils.init_distributed_mode(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    if hasattr(args, 'train_inference_task_only') and args.train_inference_task_only:
        print('Using ')
        import trainers.open_tii_trainer as tii_trainer
        tii_trainer.train(args)
    elif 'EBAG' in args.config:
        print('Using EBAG')
        import trainers.open_trainer as open_trainer
        open_trainer.train(args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    args = get_args()
    print(args)
    print(args.data_path)
    torch.cuda.empty_cache()
    main(args)