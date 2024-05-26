# Copyright (c) 2024, Guanyu Hu
# All rights reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.


import copy
import torch
import numpy as np
from fair.holisticai import multiclass_statistical_parity
from config.initializer import THRESHOLD, TrainInitializer
from fair.fairlearn.metrics import demographic_parity_difference as dpd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score


def f1_fair(y_true, y_pred, sensitive_features):
    average = 'binary' if len(np.unique(y_true)) == 2 else 'macro'
    attr_f1 = 0
    sub_attr_name_list = np.unique(sensitive_features)
    for sub_attr in sub_attr_name_list:
        subgroup_idx = sensitive_features == sub_attr
        attr_f1 += f1_score(y_true[subgroup_idx], y_pred[subgroup_idx], average=average, zero_division=0)
    return attr_f1 / len(sub_attr_name_list)


def calculate_va_ccc(label, pred):
    ccc_v = CCC_val(label[:, 0], pred[:, 0])
    ccc_a = CCC_val(label[:, 1], pred[:, 1])
    ccc_va = (ccc_v + ccc_a) / 2
    return {'ccc_v': ccc_v, 'ccc_a': ccc_a, 'ccc_va': ccc_va}


def calculate_expr_acc_f1(label, pred, average='macro'):
    acc_avg = balanced_accuracy_score(label, pred)
    f1_macro = f1_score(label, pred, average=average, zero_division=0)
    return {'acc_avg': acc_avg, 'f1_macro': f1_macro}


def calculate_au_f1(label, pred, au_len, ignore_index=None, valid_threshold=None, return_list=False):
    """

    @param label:
    @param pred:
    @param au_len:  length of AU label list, GFT_AU10, GFT_AU19
    @param ignore_index:
    @param valid_threshold:  在计算 test 数据集时候，应用 valid 最好的 threshold
    """
    # calculate f1 for each threshold
    au_f1_dict_list = []
    for idx in range(len(THRESHOLD)):
        f1_dict = {'f1': 0, 'threshold': THRESHOLD[idx]}
        if au_len > 1:
            for au in range(au_len):
                if ignore_index is not None:
                    mask = label[:, au] != -1  # 有效坐标
                    au_label = label[:, au][mask]
                    au_pred = pred[idx][:, au][mask]
                else:
                    au_label = label[:, au]
                    au_pred = pred[idx][:, au]
                f1_dict['f1'] += f1_score(au_label, au_pred, average='binary', zero_division=0)
        else:
            f1_dict['f1'] += f1_score(label, pred[idx], average='binary', zero_division=0)

        f1_dict['f1'] = f1_dict['f1'] / au_len
        au_f1_dict_list.append(f1_dict)

    # find best through each threshold
    if valid_threshold is not None:
        result_dict = {'f1_macro': -1, 'threshold': -1, 'real_f1_macro': -1, 'real_threshold': valid_threshold}
        valid_threshold_idx = THRESHOLD.index(valid_threshold)
    else:
        result_dict = {'f1_macro': -1, 'threshold': -1}

    for threshold_idx, f1_dict in enumerate(au_f1_dict_list):
        if valid_threshold is not None and threshold_idx == valid_threshold_idx:
            result_dict['real_threshold'] = valid_threshold
            result_dict['real_f1_macro'] = f1_dict['f1']
        if f1_dict['f1'] > result_dict['f1_macro']:
            result_dict['f1_macro'] = f1_dict['f1']
            result_dict['threshold'] = THRESHOLD[threshold_idx]
    if return_list:
        return {'best_dict': result_dict, 'f1_list': au_f1_dict_list}
    else:
        return result_dict


def CCC_fair(y_true, y_pred, sensitive_features):
    attr_v, attr_a = 0, 0
    sub_attr_name_list = np.unique(sensitive_features)
    for sub_attr in sub_attr_name_list:
        subgroup_idx = sensitive_features == sub_attr
        attr_v += CCC_val(y_true[:, 0][subgroup_idx], y_pred[:, 0][subgroup_idx])
        attr_a += CCC_val(y_true[:, 1][subgroup_idx], y_pred[:, 1][subgroup_idx])
    mean_v = attr_v / len(sub_attr_name_list)
    mean_a = attr_a / len(sub_attr_name_list)
    mean_va = (mean_v + mean_a) / 2
    return {'ccc_v': mean_v, 'ccc_a': mean_a, 'ccc_va': mean_va}


def calculate_fair(init, fair_methods, value, fair_result_dict, phase, result_dict=None, au_idx=None):
    au_name = None
    if init.task_type == 'AU':
        valid_idx = value['y_true'][:, au_idx] != init.ignore_index
        y_true = value['y_true'][:, au_idx][valid_idx]
        if phase == 'test':
            y_pred = value['y_pred'][THRESHOLD.index(result_dict[phase]['real_threshold'])][:, au_idx][valid_idx]
        else:
            y_pred = value['y_pred'][THRESHOLD.index(result_dict[phase][f'threshold'])][:, au_idx][valid_idx]
        au_name = init.label_index[au_idx]
        fair_dict = fair_result_dict['fair_dict'][phase]
    elif init.task_type == 'EXPR':
        valid_idx = value['y_true'] != init.ignore_index
        y_true = value['y_true'][valid_idx]
        y_pred = value['y_pred'][valid_idx]
        fair_dict = fair_result_dict['fair_mean'][phase]
    else:
        valid_idx = [True for _ in range(value['y_true'].shape[0])]
        y_true = value['y_true']
        y_pred = value['y_pred']
        fair_dict = fair_result_dict['fair_mean'][phase]

    gender_idx = value['gender'][valid_idx] != 'Unsure'
    race_idx = value['race'][valid_idx] != 'Other'
    attributes_dict = {'age': value['age'][valid_idx], 'gender': value['gender'][valid_idx][gender_idx],
                       'race': value['race'][valid_idx][race_idx]}

    for fair_method_key, fair_method in fair_methods.items():
        for attr_name, attr_value in attributes_dict.items():
            fair_dict_attr = fair_dict[attr_name]
            if fair_method_key not in fair_dict_attr.keys():
                fair_dict_attr[fair_method_key] = {}

            if attr_name == 'gender':
                fair_value = fair_method(y_true[gender_idx], y_pred[gender_idx], sensitive_features=attr_value)
            elif attr_name == 'race':
                fair_value = fair_method(y_true[race_idx], y_pred[race_idx], sensitive_features=attr_value)
            else:
                fair_value = fair_method(y_true, y_pred, sensitive_features=attr_value)

            if au_name is not None:
                fair_dict_attr[fair_method_key][au_name] = fair_value
            else:
                fair_dict_attr[fair_method_key] = fair_value

    if init.task_type == 'AU':
        for attr_name, attr_dict in fair_result_dict['fair_dict'][phase].items():
            fair_dict_attr = fair_result_dict['fair_dict'][phase][attr_name]
            fair_mean_attr = fair_result_dict['fair_mean'][phase][attr_name]
            for fair_method_key in fair_dict_attr.keys():
                fair_mean_attr[fair_method_key] = np.mean(list(fair_dict_attr[fair_method_key].values()))
    return fair_result_dict


def CCC_loss(true, pred):
    pred = pred.view(-1)
    true = true.view(-1)
    v_true = true - torch.mean(true)
    v_pred = pred - torch.mean(pred)
    rho = torch.sum(v_true * v_pred) / (
            torch.sqrt(torch.sum(torch.pow(v_true, 2))) * torch.sqrt(torch.sum(torch.pow(v_pred, 2))) + 1e-8)
    true_m = torch.mean(true)
    pred_m = torch.mean(pred)
    true_s = torch.std(true)
    pred_s = torch.std(pred)
    ccc = 2 * rho * true_s * pred_s / (torch.pow(true_s, 2) + torch.pow(pred_s, 2) + torch.pow(true_m - pred_m, 2))
    return 1 - ccc


def CCC_val(true, pred):
    true = true.astype(np.float32)
    pred = pred.astype(np.float32)
    v_true = true - true.mean()
    v_pred = pred - pred.mean()
    rho = np.sum(v_true * v_pred) / (np.sqrt(np.sum(np.power(v_true, 2))) * np.sqrt(np.sum(np.power(v_pred, 2))) + 1e-8)
    true_m = np.mean(true)
    pred_m = np.mean(pred)
    true_s = np.std(true)
    pred_s = np.std(pred)
    ccc = 2 * rho * true_s * pred_s / (np.power(true_s, 2) + np.power(pred_s, 2) + np.power(true_m - pred_m, 2))
    return ccc


def compute_CCC_loss(output, label):
    loss = CCC_loss(output[:, 0], label[:, 0]) + CCC_loss(output[:, 1], label[:, 1])
    return loss


def sp(y_true, y_pred, sensitive_features):
    return multiclass_statistical_parity(sensitive_features, y_pred, aggregation_fun="mean")


def get_loss_type(task_type):
    if task_type == 'EXPR':
        return 'CE'
    elif task_type == 'AU':
        return 'BCE'
    elif task_type == 'VA':
        return 'CCC'
    else:
        raise ValueError(f'Task type {task_type} Not Supported')


def define_loss(init: TrainInitializer):
    loss_type = get_loss_type(init.task_type) if init.loss_type is None else init.loss_type
    if loss_type == 'CE':
        if init.ignore_index is None:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=init.ignore_index)
    elif loss_type == 'BCE':
        loss_fn = torch.nn.BCELoss(reduction='mean')  # need sigmoid before
    elif loss_type == 'CCC':
        loss_fn = compute_CCC_loss
    else:
        loss_fn = None
    return loss_fn


def validation(init, valid, test, train=None):
    """
    validation and fairness
    """
    phase_dict = {'valid': valid, 'test': test}
    if train is not None:
        phase_dict['train'] = train
    result_dict = dict()
    for phase, results in phase_dict.items():
        if init.task_type == 'VA':
            result_dict[phase] = calculate_va_ccc(results['y_true'], results['y_pred'])
        elif init.task_type == 'EXPR':
            result_dict[phase] = calculate_expr_acc_f1(results['y_true'], results['y_pred'])
        elif init.task_type == 'AU':
            valid_threshold = result_dict['valid']['threshold'] if phase == 'test' else None
            result_dict[phase] = calculate_au_f1(results['y_true'], results['y_pred'], init.num_class,
                                                 ignore_index=init.ignore_index, valid_threshold=valid_threshold)
        else:
            result_dict = None

    if init.fair:
        fair_result_dict = {'fair_mean': {'test': {'age': dict(), 'gender': dict(), 'race': dict()},
                                          'valid': {'age': dict(), 'gender': dict(), 'race': dict()}}}
        if 'train' in phase_dict.keys():
            fair_result_dict['fair_mean']['train'] = copy.deepcopy(fair_result_dict['fair_mean']['test'])

        if init.task_type == "AU":
            fair_result_dict['fair_dict'] = copy.deepcopy(fair_result_dict['fair_mean'])
            fair_methods = {'dpd': dpd, 'f1': f1_fair}
            for phase, results in phase_dict.items():
                for au_idx in range(init.num_class):
                    calculate_fair(init, fair_methods, results, fair_result_dict, phase, result_dict, au_idx)
        else:
            if init.task_type == "EXPR":
                fair_methods = {'sp': sp, 'f1': f1_fair}
            elif init.task_type == "VA":
                fair_methods = {'ccc': CCC_fair}
            for phase, results in phase_dict.items():
                calculate_fair(init, fair_methods, results, fair_result_dict, phase)
    else:
        fair_result_dict = None

    return dict(sorted(result_dict.items(), reverse=True)), dict(sorted(fair_result_dict.items()))
