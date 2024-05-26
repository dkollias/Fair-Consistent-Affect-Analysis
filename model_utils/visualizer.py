# Copyright (c) 2024, Guanyu Hu
# All rights reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.


import os
from time import localtime, strftime
from tqdm import tqdm


def update_pbar(init, pbar, dataloader, batch_index):
    if len(dataloader) < init.pbar_update_num:
        if batch_index == len(dataloader) - 1:
            pbar.update(len(dataloader) % init.pbar_update_num)
    else:
        if batch_index % init.pbar_update_num == 0 and batch_index != 0:
            pbar.update(init.pbar_update_num)
        elif batch_index == len(dataloader) - 1 and batch_index % init.pbar_update_num != 0:
            pbar.update(len(dataloader) % init.pbar_update_num)




def generate_formated_print(print_data, log: str = '', indent: str = '   ') -> str:
    if isinstance(print_data, dict):
        one_level = True if sum([isinstance(each['children'], float) for each in print_data.values()]) / len(
            print_data) == 1 else False
        one_level_count = 0
        for key, value in print_data.items():
            if isinstance(value['children'], dict):
                log += indent * value['level']
                if value['is_leaf_parent']:
                    log += f'== {key} ==: '
                else:
                    log += f'== {key} ==\n'
                for i, (k, v) in enumerate(value['children'].items()):
                    if isinstance(v['children'], dict):
                        log += indent * v['level']
                        if v['is_leaf_parent']:
                            log += f'== {k} ==: '
                        else:
                            log += f'== {k} ==\n'
                        log = generate_formated_print(v['children'], log)[:-2]
                    else:
                        log += f"{k}: {float(v['children']):<9.6f}   "
                    if not isinstance(v['children'], float) or i == len(value['children']) - 1:
                        log += '\n'
            else:
                if value['level'] == 0 and (one_level_count > 4 and one_level):
                    log += f"{key}: {float(value['children']):<9.6f}\n"
                    one_level_count = 0
                else:
                    one_level_count += 1
                    log += f"{key}: {float(value['children']):<9.6f}  "
    else:
        log += f" {float(print_data['children']):<9.6f}\n"
        print(log)
    return log


def get_num_dict_log(loss_dict, topic=None, epoch=None, newline=None, break_info_line=False):
    """

    @param loss_dict:
    @param epoch:
    @param topic:
    @param newline: left, right, both, none
    @param break_info_line:
    @return:
    """
    loss_log = f'!!!!!! {topic} !!!!!!: '.upper() if topic is not None else ''
    if break_info_line:
        loss_log += '\n'
    loss_log += generate_formated_print(parse_num_dict(loss_dict))
    loss_log = get_print_message(loss_log, epoch=epoch, newline=newline)
    return loss_log


def parse_num_dict(num_dict, indent_level: int = -2, log_dict=None, upper_level_log_dict=None):
    if log_dict is None:
        log_dict = {'level': indent_level, 'is_leaf_parent': True, 'children': {}, 'is_leaf': False}
    if upper_level_log_dict is None:
        upper_level_log_dict = {'level': indent_level, 'is_leaf_parent': True, 'children': None, 'is_leaf': False}

    indent_level += 1
    if isinstance(num_dict, dict):
        upper_level_log_dict['is_leaf_parent'] = False
        log_dict['children'] = {}
        for key, value in num_dict.items():
            indent_level += 1
            log_dict['children'][key] = {'level': indent_level, 'is_leaf_parent': True, 'children': None,
                                         'is_leaf': False}
            if isinstance(value, dict):
                log_dict['is_leaf_parent'] = False
                log_dict['children'][key]['children'] = {}
                for k, v in value.items():
                    indent_level += 1
                    log_dict['children'][key]['children'][k] = {'level': indent_level, 'is_leaf_parent': True,
                                                                'children': None, 'is_leaf': False}
                    if isinstance(v, dict):
                        log_dict['children'][key]['is_leaf_parent'] = False
                        parse_num_dict(v, indent_level - 1, log_dict['children'][key]['children'][k],
                                       log_dict['children'][key])
                    else:
                        log_dict['children'][key]['children'][k]['children'] = float(v)
                        log_dict['children'][key]['children'][k]['is_leaf_parent'] = False
                        log_dict['children'][key]['children'][k]['is_leaf'] = True
                    indent_level -= 1
            else:
                log_dict['children'][key]['children'] = float(value)
                log_dict['children'][key]['is_leaf_parent'] = False
                log_dict['children'][key]['is_leaf'] = True
            indent_level -= 1
    else:
        log_dict['children'] = float(num_dict)
        log_dict['is_leaf_parent'] = False
        log_dict['is_leaf'] = True
    return log_dict['children']


def print_num_dict(loss_dict, topic=None, epoch=None, newline=None, break_info_line=False):
    """

    @param loss_dict:
    @param epoch:
    @param topic:
    @param newline: left, right, both, none
    @param break_info_line:
    @return:
    """
    num_dict_log = get_num_dict_log(loss_dict, topic, epoch, newline, break_info_line)
    tqdm.write(num_dict_log)
    return num_dict_log


def print_models(init, model, criterion=None):
    """Visualize Model"""
    tqdm.write(f'======================= Training Info =======================')
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # tqdm.write("Model = %s" % str(model))
    tqdm.write(f'=> Number of params (M): {n_parameters / 1.e6:.2f}M')
    tqdm.write(f"=> Batch size: {init.batch_size}")
    tqdm.write(f'=> The init learning rate: {init.lr}')
    if criterion is not None:
        tqdm.write(f"=> Criterion: {str(criterion)}")
    tqdm.write(f"=> Seed: = {str(init.seed)}\n")


def create_tensorboard_summarywriter(init):
    from torch.utils.tensorboard import SummaryWriter
    """ TensorBoard"""
    init.tensorboard_writer = SummaryWriter(log_dir=os.path.join(init.checkpoint_root, 'tensorboard'))


def get_print_message(message, topic=None, epoch=None, newline=None):
    """
    @param topic:
    @param epoch:
    @param message:
    @param newline: left, right, both, none
    """
    log_message = f'{strftime("%Y-%m-%d %H:%M:%S", localtime())} => '
    log_message = f'Epoch {epoch}, ' + log_message if epoch is not None else log_message
    log_message = f'!!!!!! {topic.upper()} !!!!!!  ' + log_message if topic is not None else log_message

    if newline == 'left':
        log_message = '\n' + log_message + message
    elif newline == 'right':
        log_message = log_message + message + '\n'
    elif newline == 'both':
        log_message = '\n' + log_message + message + '\n'
    else:
        log_message = log_message + message

    return log_message


def print_message(message, topic=None, epoch=None, newline=None):
    """
    @param message:
    @param epoch:
    @param newline: left, right, both, none
    """

    log_message = get_print_message(message, topic, epoch, newline)
    tqdm.write(log_message)
