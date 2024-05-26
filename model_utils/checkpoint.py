# Copyright (c) 2024, Guanyu Hu
# All rights reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.


import copy
import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from config.initializer import TrainInitializer
from model_utils.data_processing import save_pkl, remove_file_from_dir_contain_pattern
from model_utils.visualizer import print_message, print_num_dict, get_print_message


def early_stop_counter(best_dict, old_best_dict, epoch, early_stop_count, early_stop_threshold):
    sub_total = 0
    for phase in best_dict.keys():
        for key, value in best_dict[phase].items():
            if not isinstance(value, list):
                sub_total += value - old_best_dict[phase][key]

    # Early Stop
    if abs(sub_total) < early_stop_threshold:
        early_stop_count += 1
    else:
        early_stop_count = 0
    print_message(f'EarlyStop count：{early_stop_count}, Sub_total：{sub_total}', 'early stop', epoch)

    return early_stop_count


def save_txt_result(result, save_path, message=None, epoch=None):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    with open(save_path, 'a') as f:
        f.write(f'{result}\n')
    if message is not None:
        print_message(f'{message} Result Saved at: {save_path}', topic='SAVING', epoch=epoch)


def plot_loss_tensorboard(tag, loss_dict, total_step, tensorboard_writer):
    def add_scalar_recursive(prefix, sub_dict):
        for key, value in sub_dict.items():
            if isinstance(value, dict):
                add_scalar_recursive(f"{prefix}_{key}", value)
            else:
                tensorboard_writer.add_scalar(f"{tag}_{prefix}/{key}", value, total_step, new_style=True)

    if isinstance(loss_dict, dict):
        for loss_name, loss in loss_dict.items():
            if isinstance(loss, dict):
                add_scalar_recursive(loss_name, loss)
            else:
                tensorboard_writer.add_scalar(f"{tag}", loss, total_step, new_style=True)
    else:
        tensorboard_writer.add_scalar(f"{tag}", loss_dict, total_step, new_style=True)


def load_checkpoint_file(ckp, model, optimizer=None):
    tqdm.write('======================= Loading Checkpoint =======================')

    if os.path.isfile(ckp):
        checkpoint = torch.load(ckp, map_location='cuda')
        epoch = checkpoint['epoch']
        print_message(f"'{ckp}' (epoch {checkpoint['epoch']})", topic='Loading Checkpoint')
        print_message(f'Checkpoint dict keys: {list(checkpoint.keys())}', topic='Loading Checkpoint')
        try:
            if Path(ckp).name == 'mae_pretrain_vit_base':
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception:
            model_dict = model.state_dict()
            keys = []
            for k, v in checkpoint['state_dict'].items():
                keys.append(k)
            i = 0
            for k, v in model_dict.items():
                if v.size() == checkpoint['state_dict'][keys[i]].size():
                    model_dict[k] = checkpoint['state_dict'][keys[i]]
                    i = i + 1
            model.load_state_dict(model_dict)

        # Print best
        if 'best_fair_dict' in checkpoint.keys():
            print_num_dict(checkpoint['best_fair_dict'], 'BEST Fair Validation',
                           break_info_line=True)
        if 'best_dict' in checkpoint.keys():
            print_num_dict(checkpoint['best_dict'], 'BEST Validation', newline='right')
        if 'result' in checkpoint.keys():
            print_num_dict(checkpoint['result'], 'results', newline='right', break_info_line=True)
        return model, optimizer, epoch, checkpoint
    else:
        raise get_print_message(f"No checkpoint found at '{ckp}'", topic='Loading Checkpoint', newline='right')


def save_best_checkpoint(init: TrainInitializer, result, test_dict, epoch, model, optimizer,
                         output_fair_results, fair_results=None):
    os.makedirs(init.checkpoint_pth_root, exist_ok=True)

    best_dict = copy.deepcopy(init.best_dict)
    # 提前取当前checkpoint的best_dict, 避免保存时候没遍历到的best结果不对
    for phase in best_dict.keys():
        for key in best_dict[phase].keys():
            measuring_name = key.replace('best_', '')
            if 'best' in key and result[phase][measuring_name] > best_dict[phase][key]:
                best_dict[phase][key] = result[phase][measuring_name]
                if phase == 'valid':
                    output_fair_results = fair_results if fair_results is not None else None
                    real_test_name = key.replace('best', 'real')
                    if init.task_type == 'AU':
                        best_dict['test'][real_test_name] = result['test'][f'real_{measuring_name}']
                    else:
                        best_dict['test'][real_test_name] = result['test'][measuring_name]

    for key in init.best_dict['valid'].keys():
        if best_dict['valid'][key] > init.best_dict['valid'][key] and 'threshold' not in key:
            if fair_results is not None:
                if key in ['best_f1_macro', 'best_ccc_va']:
                    output_fair_results = fair_results

            if init.task_type == 'AU':
                threshold = best_dict['valid']['best_threshold']
                # remove former saved files
                inference_pattern = re.compile(
                    rf"{re.escape(init.model)}_{re.escape(init.dataset)}_{re.escape(key)}_threshold(.*)_inference_results.pkl")
                remove_file_from_dir_contain_pattern(init.checkpoint_pth_root, inference_pattern)
                ckp_pattern = re.compile(
                    rf"{re.escape(init.model)}_{re.escape(init.dataset)}_{re.escape(key)}_threshold(.*).pth")
                remove_file_from_dir_contain_pattern(init.checkpoint_pth_root, ckp_pattern)
                fair_pattern = re.compile(
                    rf"{re.escape(init.model)}_{re.escape(init.dataset)}_{re.escape(key)}_threshold(.*)_fair_results.pth")
                remove_file_from_dir_contain_pattern(init.checkpoint_pth_root, fair_pattern)

                # save path
                inference_path = init.checkpoint_pth_root.joinpath(
                    f"{init.model}_{init.dataset}_{key}_threshold{threshold}_inference_results.pkl")
                fair_path = init.checkpoint_pth_root.joinpath(
                    f"{init.model}_{init.dataset}_{key}_threshold{threshold}_fair_results.pkl")
                ckp_path = init.checkpoint_pth_root.joinpath(
                    f"{init.model}_{init.dataset}_{key}_threshold{threshold}.pth")
            else:
                inference_path = init.checkpoint_pth_root.joinpath(
                    f"{init.model}_{init.dataset}_{key}_inference_results.pkl")
                fair_path = init.checkpoint_pth_root.joinpath(
                    f"{init.model}_{init.dataset}_{key}_fair_results.pkl")
                ckp_path = init.checkpoint_pth_root.joinpath(f"{init.model}_{init.dataset}_{key}.pth")

            """ save fair results """
            save_pkl(output_fair_results, fair_path)
            """ save eval results """
            save_pkl(test_dict, inference_path)
            """ save checkpoint """
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_dict': best_dict,
                        'fair_dict': output_fair_results,
                        'result': result}, ckp_path)

    init.best_dict = best_dict
    print_message(f'Checkpoint and Inference Results Saved at: {init.checkpoint_pth_root}', topic='Checkpoint',
                  epoch=epoch)

    if init.result_to_table:
        """ Results Table """
        full_dict = {}
        concise_dict = {}
        for phase, phase_dict in best_dict.items():
            if phase == 'train':
                continue
            for best_key, value in phase_dict.items():
                full_dict[f'best_{phase}_{best_key}'] = value
                if phase != 'valid' and 'real' in best_key:
                    if init.task_type == 'VA':
                        if best_key == 'real_ccc_va':
                            concise_dict[f'best_{phase}_{best_key}'] = value
                    else:
                        concise_dict[f'best_{phase}_{best_key}'] = value

        for phase, phase_dict in output_fair_results['fair_mean'].items():
            if phase == 'train':
                continue
            for demographic_key, demographic_dict in phase_dict.items():
                for metric_key, metric_value in demographic_dict.items():
                    if init.task_type == 'VA':
                        for key, value in metric_value.items():
                            full_dict[f'fair_{phase}_{demographic_key}_{metric_key}_{key}'] = value
                    else:
                        full_dict[f'fair_{phase}_{demographic_key}_{metric_key}'] = metric_value
                    if phase != 'valid':
                        if init.task_type == 'VA':
                            for key, value in metric_value.items():
                                if key == 'ccc_va':
                                    concise_dict[f'fair_{phase}_{demographic_key}_{metric_key}_{key}'] = value
                        else:
                            concise_dict[f'fair_{phase}_{demographic_key}_{metric_key}'] = metric_value

        full_table_save_path = init.checkpoint_root.joinpath(f"{init.model}_{init.dataset}_full_table.csv")
        concise_table_save_path = init.checkpoint_root.joinpath(f"{init.model}_{init.dataset}_concise_table.csv")
        pd.DataFrame({init.model: dict(sorted(concise_dict.items()))}).T.to_csv(concise_table_save_path)
        pd.DataFrame({init.model: dict(sorted(full_dict.items()))}).T.to_csv(full_table_save_path)

    with open(init.best_result_txt_path, "w") as f:
        if output_fair_results is not None:
            best_fair_result_log = print_num_dict(output_fair_results, 'BEST Fair Validation', epoch,
                                                  break_info_line=True, newline='left')
            f.write(best_fair_result_log)
            print_message(f'Best Results & Fair Results saved at: {init.best_result_txt_path}',
                          topic='Saving Checkpoint', epoch=epoch)
        else:
            print_message(f'Best Results saved at: {init.best_result_txt_path}', topic='Saving Checkpoint', epoch=epoch)
        best_result_log = print_num_dict(best_dict, 'BEST Validation', epoch, break_info_line=True)
        f.write(best_result_log)

    return best_dict, output_fair_results
