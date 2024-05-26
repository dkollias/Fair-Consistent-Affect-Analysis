# Copyright (c) 2024, Guanyu Hu
# All rights reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

import copy
import time
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from config.initializer import TrainInitializer
from model_utils.dataloader import load_image_data
from model_utils.validation import define_loss, validation
from model_utils.training import train_model, eval_model, get_lr
from model_utils.visualizer import create_tensorboard_summarywriter
from model_utils.checkpoint import save_best_checkpoint, early_stop_counter
from model_utils.checkpoint import load_checkpoint_file, save_txt_result, plot_loss_tensorboard
from model_utils.visualizer import print_models, print_num_dict, print_message, get_num_dict_log


def main(init: TrainInitializer):
    device = init.device
    epoch, early_stop_count = -1, 0
    output_fair_results = None

    """ DataLoader """
    train_dataloader, val_dataloader, test_dataloader = load_image_data(init)

    """ Model """
    model = init.get_model(init)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=init.lr, weight_decay=0.0001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = define_loss(init)

    """ Visualize Model """
    print_models(init, model, criterion)

    """ Load & Print Best Checkpoint """
    if init.loaded_checkpoint_path is not None:
        model, optimizer, epoch, checkpoint = load_checkpoint_file(init.loaded_checkpoint_path, model, optimizer)

        if init.only_print_ckp_best:
            return  # break when only print

        """ Run Evaluation """
        valid_dict = eval_model(init, 'valid', model, val_dataloader, criterion, device)
        test_dict = eval_model(init, 'test', model, test_dataloader, criterion, device)
        best_dict, output_fair_results = validation(init, valid_dict, test_dict)

        result_log = get_num_dict_log(best_dict, topic='Validation', epoch=epoch)
        fair_result_log = get_num_dict_log(output_fair_results, topic='Fair Validation', epoch=epoch, break_info_line=True)
        save_txt_result(result_log, init.result_txt_path, message='Validation', epoch=epoch)
        save_txt_result(fair_result_log, init.fair_result_txt_path, message='Fair Validation', epoch=epoch)
        if output_fair_results is not None:
            print_num_dict(output_fair_results, 'BEST Fair Validation', epoch, break_info_line=True)
        print_num_dict(best_dict, 'BEST Validation', epoch=epoch, newline='right', break_info_line=True)
        return

    """ Training Pipeline """
    create_tensorboard_summarywriter(init)
    tqdm.write(f'======================= Start Training =======================')
    for epoch in range(1, init.epochs + 1):
        tic = time.time()
        tqdm.write(f'\n=> Epoch: {epoch}/{init.epochs}, Current lr: {get_lr(optimizer)}')
        train_dict = train_model(init, model, train_dataloader, criterion, optimizer, device)
        valid_dict = eval_model(init, 'valid', model, val_dataloader, criterion, device)
        test_dict = eval_model(init, 'test', model, test_dataloader, criterion, device)
        scheduler.step()
        toc = time.time()
        print_message(f"Epoch {epoch} Using Time: {toc - tic:.3f}s", topic='TIME USING', epoch=epoch)

        """ Visualize Loss """
        loss_dict = {'train': train_dict['loss'], 'valid': valid_dict['loss'], 'test': test_dict['loss']}
        loss_log = print_num_dict(loss_dict, topic='Loss', epoch=epoch, newline='both')
        save_txt_result(loss_log, init.loss_txt_path, message='Loss', epoch=epoch)
        plot_loss_tensorboard('Loss', loss_dict, epoch, init.tensorboard_writer)

        """ Validation Results """
        results, fair_results = validation(init, valid_dict, test_dict, train=train_dict)
        result_log = get_num_dict_log(results, topic='Validation', epoch=epoch)
        fair_result_log = get_num_dict_log(fair_results, topic='Fair Validation', epoch=epoch, break_info_line=True)
        save_txt_result(result_log, init.result_txt_path, message='Validation', epoch=epoch)
        save_txt_result(fair_result_log, init.fair_result_txt_path, message='Fair Validation', epoch=epoch)
        plot_loss_tensorboard('Validation', results, epoch, init.tensorboard_writer)

        """ Save Best """
        old_best_dict = copy.deepcopy(init.best_dict)  # initializer.best_dict
        best_dict, output_fair_results = save_best_checkpoint(init, results, test_dict, epoch, model, optimizer,
                                                              output_fair_results, fair_results)
        early_stop_count = early_stop_counter(best_dict, old_best_dict, epoch, early_stop_count,
                                              init.early_stop_threshold)
        if early_stop_count >= init.early_stop_round:
            if output_fair_results is not None:
                print_num_dict(output_fair_results, 'BEST Fair Validation', epoch, break_info_line=True, newline='left')
            print_num_dict(best_dict, 'BEST Validation', epoch, newline='right', break_info_line=True)
            break

    if output_fair_results is not None:
        print_num_dict(output_fair_results, 'BEST Fair Validation', epoch, break_info_line=True, newline='left')
    print_num_dict(init.best_dict, 'BEST Validation', epoch, newline='right', break_info_line=True)
    print_message(f'End Training at {epoch}!!!', topic='Ending', epoch=epoch)
