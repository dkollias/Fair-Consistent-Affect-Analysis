# Copyright (c) 2024, Guanyu Hu
# All rights reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.


import torch
import numpy as np
from tqdm import tqdm
from model_utils.visualizer import update_pbar
from config.initializer import TrainInitializer, THRESHOLD, BaseInitializer


def process_batch(batch, device, age_list=None, gender_list=None, race_list=None, path_list=None, fair=True):
    if fair:
        imgs, labels, age, gender, race, path = batch

        age_list.append(age)
        gender_list.append(gender)
        race_list.append(race)

        path_list.append(path)

        labels = labels.to(device)
        imgs = imgs.to(device)
        return imgs, labels, age, gender, race
    else:
        imgs, labels, path = batch
        path_list.append(path)
        labels = labels.to(device)
        imgs = imgs.to(device)
        return imgs, labels


def get_label_pred_array(init: BaseInitializer, running_loss, y_true, y_pred, age_list, gender_list, race_list,
                         image_file_path):
    y_true_array = np.concatenate(y_true)
    image_file_path_array = np.concatenate(image_file_path)

    if init.task_type == 'AU':
        y_pred_array = list()
        for index, _ in enumerate(y_pred):
            y_pred_array.append(np.concatenate(y_pred[index]))
    else:  # Expr, VA
        y_pred_array = np.concatenate(y_pred)

    if init.fair:
        age_array = np.concatenate(age_list)
        gender_array = np.concatenate(gender_list)
        race_array = np.concatenate(race_list)
        return {'loss': running_loss, 'y_pred': y_pred_array, 'y_true': y_true_array,
                'image_file_path': image_file_path_array, 'age': age_array, 'gender': gender_array, 'race': race_array}
    else:
        return {'loss': running_loss, 'y_pred': y_pred_array, 'y_true': y_true_array,
                'image_file_path': image_file_path_array}


def append_expr_label_pred_to_list(outputs, labels, y_pred_list, y_true_list):
    _, predicts = torch.max(outputs, 1)
    y_pred_list.append(predicts.cpu().numpy())
    if len(labels.shape) == 1:
        real = labels
    elif len(labels.shape) == 2:
        _, real = torch.max(labels, 1)
    else:
        raise ValueError('labels shape error')
    y_true_list.append(real.cpu().numpy())


def append_va_label_pred_to_list(outputs, labels, y_pred_list, y_true_list):
    y_pred_list.append(outputs.detach().cpu().numpy())
    y_true_list.append(np.array(labels.cpu()))


def append_au_label_pred_to_list(outputs, labels, y_pred_list, y_true_list):
    # Pred List
    for i, thresh in enumerate(THRESHOLD):
        pred = outputs >= thresh
        pred = pred.long()
        if len(y_pred_list) < len(THRESHOLD):
            y_pred_list.append(list())
        y_pred_list[i].append(pred.cpu().numpy())
    # True List
    y_true_list.append(labels.cpu().numpy())


def base_append_label_pred_to_list(init: BaseInitializer, outputs, labels, y_pred_list, y_true_list):
    """ Label, Prediction """
    if init.task_type == 'EXPR':
        append_expr_label_pred_to_list(outputs, labels, y_pred_list, y_true_list)
    elif init.task_type == 'VA':
        append_va_label_pred_to_list(outputs, labels, y_pred_list, y_true_list)
    elif init.task_type == 'AU':
        append_au_label_pred_to_list(outputs, labels, y_pred_list, y_true_list)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(init: TrainInitializer, model, train_dataloader, criterion, optimizer, device):
    tqdm.write('======= Running Train =======')
    model.train()
    model.to(device)
    running_loss = 0
    y_true, y_pred = list(), list()
    age_list, gender_list, race_list = list(), list(), list()
    image_file_path = list()

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for batch_index, batch in enumerate(train_dataloader):
        """ Data """
        if init.fair:
            imgs, labels, age, gender, race = init.process_batch(batch, device, age_list, gender_list, race_list,
                                                                 image_file_path, fair=True)
        else:
            imgs, labels = init.process_batch(batch, device, path_list=image_file_path, fair=False)

        optimizer.zero_grad()

        """ Training """
        if init.use_sigmoid:
            outputs = model(imgs).sigmoid()
        else:
            outputs = model(imgs)

        """ Loss """
        if init.task_type == 'EXPR':
            loss = criterion(outputs, labels.long())
        else:  # VA, AU
            if init.ignore_index is not None and init.loss_type != 'WeightedAsymmetricLoss':
                mask = labels != init.ignore_index
                loss = criterion(outputs[mask], labels.to(torch.float)[mask])
            else:
                loss = criterion(outputs, labels.to(torch.float))

        loss.backward()
        optimizer.step()
        running_loss += loss

        """ Append Label & Prediction to List"""
        base_append_label_pred_to_list(init, outputs, labels, y_pred, y_true)

        """ tqdm """
        update_pbar(init, pbar, train_dataloader, batch_index)
    pbar.close()

    """ Loss """
    running_loss = running_loss / len(train_dataloader)

    """ Return Label & Prediction Array """
    return get_label_pred_array(init, running_loss, y_true, y_pred, age_list, gender_list, race_list, image_file_path)


def eval_model(init: TrainInitializer, phase, model, test_dataloader, criterion, device):
    tqdm.write('======= Running Evaluate =======')

    model.eval()
    running_loss = 0.0
    y_true, y_pred = list(), list()
    age_list, gender_list, race_list = list(), list(), list()
    image_file_path = list()

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f'{phase}: ', position=0, leave=True)
    for batch_index, batch in enumerate(test_dataloader):
        """ Data """
        if init.fair:
            imgs, labels, age, gender, race = init.process_batch(batch, device, age_list, gender_list,
                                                                 race_list, image_file_path, fair=True)
        else:
            imgs, labels = init.process_batch(batch, device, path_list=image_file_path, fair=False)

        """ Inference """
        with torch.no_grad():
            if init.use_sigmoid:
                outputs = model(imgs).sigmoid()
            else:
                outputs = model(imgs)

        """ Loss """
        if init.task_type == 'EXPR':
            loss = criterion(outputs, labels.long())
        else:  # VA, AU
            if init.ignore_index is not None and init.loss_type != 'WeightedAsymmetricLoss':
                mask = labels != init.ignore_index
                loss = criterion(outputs[mask], labels.to(torch.float)[mask])
            else:
                loss = criterion(outputs, labels.to(torch.float))
        running_loss += loss

        """ Append Label & Prediction to List"""
        base_append_label_pred_to_list(init, outputs, labels, y_pred, y_true)

        """ tqdm """
        update_pbar(init, pbar, test_dataloader, batch_index)
    pbar.close()

    """ Loss """
    running_loss = running_loss / len(test_dataloader)

    """ Return Label & Prediction Array """
    return get_label_pred_array(init, running_loss, y_true, y_pred, age_list, gender_list, race_list, image_file_path)
