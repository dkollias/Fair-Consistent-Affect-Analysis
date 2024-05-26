# Copyright (c) 2024, Guanyu Hu
# All rights reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

import os
import pickle
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
import yaml

def load_yaml(file_path):
    yaml.add_constructor('!join_str', join_str)
    yaml.add_constructor('!join_path', join_path)
    with open(file_path) as stream:
        yaml_data = yaml.load(stream, Loader=yaml.Loader)
    return yaml_data


def join_str(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def join_path(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.join(*[str(i) for i in seq])


def save_pkl(save_object, save_dir, file_name=None, verbose=False):
    if Path(save_dir).suffix in ['.pkl', '.pickle']:
        save_path = save_dir
    else:
        save_path = os.path.join(save_dir, f'{file_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_object, f)
    if verbose:
        tqdm.write("=> Pickle File Saved at {}".format(save_path))
    return save_path


def remove_file_from_dir_contain_pattern(dir_root, pattern):
    file_list = list_file_abs_path(dir_root)
    for file_path in file_list:
        if pattern.search(file_path):
            os.remove(file_path)
            tqdm.write(f"Deleted file: {file_path}")


def list_file_abs_path(file_dir, path_remove_content=None, sort=True, verbose=False):
    """

    :param file_dir: 目录地址
    :return: 返回 file_dir 目录下的所有【子文件】的绝对地址，且排序
    """
    if check_dir_exist(file_dir, verbose=verbose):
        root, dirs, files = next(os.walk(file_dir))
        file_abs_path_list = [os.path.join(root, f) for f in files]
        if path_remove_content is not None:
            if path_remove_content[-1] != '/':
                path_remove_content += '/'
            new_list = list()
            for each in file_abs_path_list:
                new_list.append(each.replace(path_remove_content, ''))
            file_abs_path_list = new_list
        if verbose:
            print(f'======================= Total Files: {len(file_abs_path_list)} =============================')
        if sort:
            return natsorted(file_abs_path_list, key=str)
        else:
            return file_abs_path_list


# 检查目录是否存在，若存在则 return True
def check_dir_exist(dir_path: str, exception=True, verbose=True):
    """
    检查目录是否存在，若存在则 return True

    :param dir_path: 目录路径
    :param exception: 当 目录不存在 时是否报警

    :return: check_output: 若存在则 return True
    """

    if os.path.isdir(dir_path):
        return True
    else:
        error_msg = f'!!!!!!!!!!! Dir "{dir_path}" not exist! !!!!!!!!!!!'
        alert_exception(error_msg, exception, verbose)
        return False


def alert_exception(error, is_exception=True, verbose=True):
    if is_exception:
        raise Exception(error)
    else:
        if verbose:
            tqdm.write(error)
