import argparse
import random
import numpy as np
import utils
import warnings
import torch

from pathlib import Path

warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def get_args():
    # parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    # config = parser.parse_known_args()[-1][0]
    # subparser = parser.add_subparsers(dest='subparser_name')

    # if config == 'cifar100_hideprompt_5e':
    #     from configs.cifar100_hideprompt_5e import get_args_parser
    #     config_parser = subparser.add_parser('cifar100_hideprompt_5e', help='Split-CIFAR100 HiDe-Prompt configs')
    # elif config == 'cifar100_norgaprompt':
    #     from configs.cifar100_norgaprompt import get_args_parser
    #     config_parser = subparser.add_parser('cifar100_norgaprompt', help='Split-CIFAR100 EBAG configs')
    # elif config == 'imr_hideprompt_5e':
    #     from configs.imr_hideprompt_5e import get_args_parser
    #     config_parser = subparser.add_parser('imr_hideprompt_5e', help='Split-ImageNet-R HiDe-Prompt configs')
    # elif config == 'imr_norgaprompt':
    #     from configs.imr_norgaprompt import get_args_parser
    #     config_parser = subparser.add_parser('imr_norgaprompt', help='Split-ImageNet-R EBAG configs')
    # elif config == 'five_datasets_hideprompt_5e':
    #     from configs.five_datasets_hideprompt_5e import get_args_parser
    #     config_parser = subparser.add_parser('five_datasets_hideprompt_5e', help='five datasets HiDe-Prompt configs')
    # elif config == 'five_datasets_norgaprompt':
    #     from configs.five_datasets_norgaprompt import get_args_parser
    #     config_parser = subparser.add_parser('five_datasets_norgaprompt', help='five datasets EBAG configs')
    # elif config == 'cub_hideprompt_5e':
    #     from configs.cub_hideprompt_5e import get_args_parser
    #     config_parser = subparser.add_parser('cub_hideprompt_5e', help='Split-CUB HiDe-Prompt configs')
    # elif config == "cub_norgaprompt":
    #     from configs.cub_norgaprompt import get_args_parser
    #     config_parser = subparser.add_parser('cub_norgaprompt', help='Split-CUB EBAG configs')
    # elif config == 'cifar100_dualprompt':
    #     from configs.cifar100_dualprompt import get_args_parser
    #     config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 dual-prompt configs')
    # elif config == 'imr_dualprompt':
    #     from configs.imr_dualprompt import get_args_parser
    #     config_parser = subparser.add_parser('imr_dualprompt', help='Split-ImageNet-R dual-prompt configs')
    # elif config == 'five_datasets_dualprompt':
    #     from configs.five_datasets_dualprompt import get_args_parser
    #     config_parser = subparser.add_parser('five_datasets_dualprompt', help='five datasets dual-prompt configs')
    # elif config == 'cub_dualprompt':
    #     from configs.cub_dualprompt import get_args_parser
    #     config_parser = subparser.add_parser('cub_dualprompt', help='Split-CUB dual-prompt configs')
    # elif config == 'cifar100_sprompt_5e':
    #     from configs.cifar100_sprompt_5e import get_args_parser
    #     config_parser = subparser.add_parser('cifar100_sprompt_5e', help='Split-CIFAR100 s-prompt configs')
    # elif config == 'imr_sprompt_5e':
    #     from configs.imr_sprompt_5e import get_args_parser
    #     config_parser = subparser.add_parser('imr_sprompt_5e', help='Split-ImageNet-R s-prompt configs')
    # elif config == 'five_datasets_sprompt_5e':
    #     from configs.five_datasets_sprompt_5e import get_args_parser
    #     config_parser = subparser.add_parser('five_datasets_sprompt_5e', help='five datasets s-prompt configs')
    # elif config == 'cub_sprompt_5e':
    #     from configs.cub_sprompt_5e import get_args_parser
    #     config_parser = subparser.add_parser('cub_sprompt_5e', help='Split-CUB s-prompt configs')
    # elif config == 'cifar100_l2p':
    #     from configs.cifar100_l2p import get_args_parser
    #     config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 l2p configs')
    # elif config == 'imr_l2p':
    #     from configs.imr_l2p import get_args_parser
    #     config_parser = subparser.add_parser('imr_l2p', help='Split-ImageNet-R l2p configs')
    # elif config == 'five_datasets_l2p':
    #     from configs.five_datasets_l2p import get_args_parser
    #     config_parser = subparser.add_parser('five_datasets_l2p', help='five datasets l2p configs')
    # elif config == 'cub_l2p':
    #     from configs.cub_l2p import get_args_parser
    #     config_parser = subparser.add_parser('cub_l2p', help='Split-CUB l2p configs')
    # else:
    #     raise NotImplementedError
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    # config = 'imr_norgaprompt'
    # config = 'cifar100_norgaprompt'
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