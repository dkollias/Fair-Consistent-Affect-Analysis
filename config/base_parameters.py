# Copyright (c) 2024, Guanyu Hu
# All rights reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.


import argparse
from tqdm import tqdm



def get_args_parser(parser_list: list):
    args_parser = argparse.ArgumentParser(parents=parser_list)
    known_args, unknown_args = args_parser.parse_known_args()
    tqdm.write(f'=> !!!!!!!!!!!!!!!!!!! UnKnown args: {unknown_args} !!!!!!!!!!!!!!!!!!!')
    return known_args

def base_parser():
    """ Basic Parameters """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--result_to_table', default=True, action='store_true',
                        help='Write experiment results to the summary table.')

    parser.add_argument('--fair', default=False, action='store_true', help='Run fair validation.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes.')
    parser.add_argument('--seed', type=int, default=999, help='Random seed.')

    """ Dataset, Model, Batch Size """
    parser.add_argument('--yaml_path', type=str, default='config/config.yaml', help='Path to the config file.')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Name of the dataset (e.g., AffectNet-7, AffectNet-8, RAF-DB).')
    parser.add_argument('--model', type=str, default=None, help='Name of the model (e.g., resnet, vgg16, vit_b_16).')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size.')

    """ Resume """
    parser.add_argument('--eval', action='store_true', help='Evaluation mode.')
    parser.add_argument('-ckp', '--checkpoint_path', default=None, help='Path to the checkpoint file (.pkl) or None.')
    parser.add_argument('--only_print_ckp_best', action='store_true',
                        help='Print only the best result saved in the checkpoint.')

    """ Transform """
    parser.add_argument('--resize', type=int, default=224, help='Resize dimensions for the input image.')
    parser.add_argument('--augmentation', action='store_true', help='Apply data augmentation.')
    parser.add_argument('--rotation', type=int, default=30,
                        help='Rotation angle for augmentation (applied only if augmentation is true).')
    return parser


def base_train_parser():
    """Train Parameters"""
    parser = argparse.ArgumentParser(add_help=False)
    """Train"""
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('-loss', '--loss_type', default=None,
                        help="Specify the loss type (optional).This will overwrite the initializer's default loss.")
    parser.add_argument('--early_stop_round', type=int, default=20,
                        help='Number of epochs to run before early stopping if no improvement')
    parser.add_argument('--early_stop_threshold', type=float, default=0.00001,
                        help='Threshold for improvement to consider for early stopping')
    parser.add_argument('--pbar_update_num', type=int, default=100, help='Number of batches to update progress bar')
    return parser
