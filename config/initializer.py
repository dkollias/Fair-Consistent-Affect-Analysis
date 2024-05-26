# Copyright (c) 2024, Guanyu Hu
# All rights reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.


from pathlib import Path
from pprint import pformat
from time import localtime, strftime
from tqdm import tqdm
import datetime
import random, os
import numpy as np
import torch
from model_utils import data_processing

THRESHOLD = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def parse_ckp_model(dataset, loaded_checkpoint_path):
    """ return model name of ckp """
    return Path(loaded_checkpoint_path).stem.split(f'_{dataset}')[0]


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_ckp_name(dataset_name, model, timestamp: datetime.datetime = None, lr=None, batch_size=None,
                 augmentation=False):
    checkpoint_name = f'{dataset_name}-{model}'
    if timestamp is not None:
        checkpoint_name += f'-{timestamp.strftime("[%m%d-%H%M]")}'
    if lr is not None:
        checkpoint_name += f'-lr{lr}'
    if batch_size is not None:
        checkpoint_name += f'-bs{batch_size}'
    if augmentation:
        checkpoint_name += f'-aug'
    return checkpoint_name


def get_paths(dataset, config):
    try:
        dataset_root = config[f'{dataset}_dataset_root']
        csv_path_dict = {'train': os.path.join(config[f'{dataset}_csv_root'], 'train.csv'),
                         'valid': os.path.join(config[f'{dataset}_csv_root'], 'valid.csv'),
                         'test': os.path.join(config[f'{dataset}_csv_root'], 'test.csv')}
        num_class = config[f'{dataset}_num_class']
        return dataset_root, csv_path_dict, num_class
    except KeyError:
        tqdm.write(f'=> !!!!!!!!!!!!!!!! Pass get_paths in initializer for {dataset} !!!!!!!!!!!!!!!!')
        return None, None, None


def get_task(dataset):
    if dataset in ['AffectNet-7', 'AffectNet-8''RAF-DB']:
        return 'EXPR'
    elif dataset in ['DISFA', 'EmotioNet', 'GFT', 'RAF-AU']:
        return 'AU'
    elif dataset in ['AffectNet-VA']:
        return 'VA'
    else:
        raise ValueError(f'Dataset {dataset} not supported')


def get_model_name(model):
    if model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        return 'resnet'
    elif model in ['resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d']:
        return 'resnext'
    elif model in ['swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b']:
        return 'swin'
    elif model in ['vgg11', 'vgg16', 'vgg19']:
        return 'vgg'
    elif model in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']:
        return 'vit'
    elif model in ['iresnet']:
        return 'iresnet'
    elif model in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b6',
                   'efficientnet_b7', 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l']:
        return 'efficientnet'
    elif model in ['densenet121', 'densenet161', 'densenet201']:
        return 'densenet'
    elif model in ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']:
        return 'convnext'
    elif model == 'vit_base_patch16':
        return 'MAE'
    elif model == 'abaw5_ctc':
        return 'CTC'
    elif model == 'abaw5_situ':
        return 'SITU'
    elif model == 'EAC':
        return 'EAC'
    else:
        raise ValueError(f'Model {model} not supported')


class BaseInitializer:
    def __init__(self, parser):
        tqdm.write(f'\n\n=========== Program Starting at: {strftime("%Y-%m-%d %H:%M:%S", localtime())} ===========\n')

        """ basic param """
        self.now = datetime.datetime.now()
        self.seed = parser.seed
        seed_everything(parser.seed)
        self.batch_size = parser.batch_size
        self.num_workers = parser.num_workers
        self.result_to_table = parser.result_to_table

        """ Path """
        path_config = data_processing.load_yaml(parser.yaml_path)
        self.output_root = path_config['output_root']

        """ Dataset """
        self.dataset = parser.dataset
        self.task_type = get_task(self.dataset)
        self.dataset_root, self.csv_path_dict, self.num_class = get_paths(self.dataset, path_config)

        """ Fair """
        self.fair = parser.fair

        """ ignore_index """
        self.ignore_index = -1 if self.dataset == 'EmotioNet' else None

        """Transform"""
        self.resize = parser.resize
        self.rotation = parser.rotation
        self.augmentation = parser.augmentation

        """ model & checkpoint """
        self.eval = parser.eval
        if parser.checkpoint_path is not None and parser.eval:
            self.loaded_checkpoint_path = parser.checkpoint_path
            self.model = parse_ckp_model(self.dataset, parser.checkpoint_path)
        else:
            self.loaded_checkpoint_path = None
            self.model = parser.model

        self.model_name = get_model_name(self.model)
        self.only_print_ckp_best = parser.only_print_ckp_best
        self.use_sigmoid = True if self.task_type == 'AU' else False  # sigmoid only when task is AU

    def __call__(self):
        self.define_base_parameters()
        self.print_initializer()
        return self

    def define_base_parameters(self):
        """
        Define DataFolder, Batch process method, best_dict
        """
        self.define_data_folder()
        self.define_process_batch_method()  # MT
        self.get_best_dict()
        return self

    def define_data_folder(self):
        if self.model_name in ['resnet', 'resnext', 'swin', 'vgg', 'vit', 'iresnet', 'efficientnet', 'densenet',
                               'convnext']:
            from model_utils.dataloader import ImageFolder
            self.dataloader = ImageFolder
        elif self.model_name == 'CTC':
            from model_utils.ctc_model_utils.ctc_dataloader import CTCImageFolder
            self.dataloader = CTCImageFolder
        elif self.model_name in ['MAE', 'SITU']:
            from model_utils.mae_model_utils.mae_dataloader import MAEImageFolder
            self.dataloader = MAEImageFolder
        else:
            raise ValueError(f'Define_get_model_method Not Support Model {self.model_name} ')

    def define_process_batch_method(self):
        if self.model_name in ['resnet', 'resnext', 'swin', 'vgg', 'vit', 'iresnet', 'efficientnet', 'densenet',
                               'convnext']:
            from model_utils.training import process_batch
            self.process_batch = process_batch
        elif self.model_name == 'CTC':
            from model_utils.ctc_model_utils.ctc_training import ctc_process_batch
            self.process_batch = ctc_process_batch
        elif self.model_name in ['MAE', 'SITU']:
            from model_utils.training import process_batch
            self.process_batch = process_batch
        else:
            raise ValueError(f'Define_get_model_method Not Support Model {self.model_name} ')

    def get_best_dict(self):
        if self.task_type == 'EXPR':
            self.best_dict = {'valid': {'best_acc_avg': -1, 'best_f1_macro': -1},
                              'test': {'real_acc_avg': -1, 'real_f1_macro': -1,
                                       'best_acc_avg': -1, 'best_f1_macro': -1}}
        elif self.task_type == 'AU':
            self.best_dict = {'valid': {'best_f1_macro': -1, 'best_threshold': -1},
                              'test': {'real_f1_macro': -1, 'real_threshold': -1, 'best_f1_macro': -1,
                                       'best_threshold': -1}}
        elif self.task_type == 'VA':
            self.best_dict = {'valid': {'best_ccc_va': -1, 'best_ccc_v': -1, 'best_ccc_a': -1},
                              'test': {'real_ccc_v': -1, 'real_ccc_a': -1, 'real_ccc_va': -1,
                                       'best_ccc_v': -1, 'best_ccc_a': -1, 'best_ccc_va': -1}}
        else:
            raise ValueError(f'Task type {self.task_type} not supported')

    def print_initializer(self):
        tqdm.write(f"=======================  Initializer: =======================  \n{pformat(self.__dict__)}\n\n")


class TrainInitializer(BaseInitializer):
    def __init__(self, parser):
        super().__init__(parser)
        """ loss """
        self.loss_type = None if parser.loss_type is None else parser.loss_type

        """Model Training Parameters"""
        self.device = torch.device('cuda')
        self.lr = parser.lr
        self.epochs = parser.epochs
        self.early_stop_round = parser.early_stop_round
        self.early_stop_threshold = parser.early_stop_threshold
        self.pbar_update_num = parser.pbar_update_num

    def __call__(self):
        self.define_base_parameters()
        self.define_model_method()  # MT overwrite
        self.get_save_dir(self.now)
        self.print_initializer()
        return self

    def get_save_dir(self, now):
        """ Save Dir """
        if self.eval:
            self.output_dir = Path(
                *[self.output_root, 'new_output_eval', self.task_type, self.dataset, self.model_name])
        else:
            self.output_dir = Path(*[self.output_root, 'new_output', self.task_type, self.dataset, self.model_name])

        checkpoint_name = get_ckp_name(self.dataset, self.model, now, self.lr, self.batch_size, self.augmentation)
        self.checkpoint_root = self.output_dir.joinpath(checkpoint_name)
        self.checkpoint_pth_root = self.checkpoint_root.joinpath('checkpoint')
        self.loss_txt_path = self.checkpoint_root.joinpath('loss.txt')
        self.result_txt_path = self.checkpoint_root.joinpath('result.txt')
        self.fair_result_txt_path = self.checkpoint_root.joinpath('fair_result.txt')
        self.best_result_txt_path = self.checkpoint_root.joinpath('best_result.txt')

    def define_model_method(self):
        if self.model_name in ['resnet', 'resnext', 'swin', 'vgg', 'vit', 'iresnet', 'efficientnet', 'densenet',
                               'convnext']:
            from model_utils.model import get_gy_model
            self.get_model = get_gy_model
        elif self.model_name == 'MAE':
            from model_utils.mae_model_utils.mae_model import get_mae_model
            self.get_model = get_mae_model
        elif self.model_name == 'CTC':
            from model_utils.ctc_model_utils.ctc_model import get_ctc_model
            self.get_model = get_ctc_model
        elif self.model_name == 'SITU':
            from model_utils.situ_model_utils.situ_model import get_situ_model
            self.get_model = get_situ_model
        else:
            raise ValueError(f'Define_get_model_method Not Support Model {self.model_name} ')
