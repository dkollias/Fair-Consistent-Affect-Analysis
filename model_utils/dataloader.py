# Copyright (c) 2024, Guanyu Hu
# All rights reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.


import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.utils.data as data
from torch.utils.data import DataLoader as DL
from torchvision.transforms import transforms as T
from config.initializer import TrainInitializer, BaseInitializer

VIT_LIST = ['vit_h_14']


def compose_transform(transform):
    tqdm.write(f'=> Transformation: {transform}\n')
    return T.Compose(transform)


def base_transform_define(init: TrainInitializer, phase=None):
    """
     phase: train, valid==test
    """
    transform = []
    # image->initial_resize->initializer.resize
    initial_resize = 600 if init.model in VIT_LIST else 256
    if init.model == 'vit':
        initial_resize = 128
        init.resize = 112
    if init.model in ['mobileface', 'irse50'] or init.model_name == 'insightface':
        initial_resize = 128
        init.resize = 112

    """ Transforms """
    resize = T.Resize([initial_resize])
    center_crop = T.CenterCrop(init.resize)
    random_resize_crop = T.RandomResizedCrop((init.resize, init.resize))
    to_tensor = T.PILToTensor()
    to_float = T.ConvertImageDtype(torch.float)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    augmentation = [T.RandomHorizontalFlip(),
                    T.ColorJitter(),
                    T.RandomRotation(init.rotation)]  # 水平翻转,颜色修改，随机旋转

    if phase == 'train':
        transform.append(resize)
        transform.append(center_crop)
        if init.augmentation:
            transform.extend(augmentation)
        transform.append(to_tensor)
        transform.append(to_float)
        transform.append(normalize)
    else:
        transform.append(resize)
        transform.append(center_crop)
        transform.append(to_tensor)
        transform.append(to_float)
        transform.append(normalize)

    return transform


def load_image_data(init: BaseInitializer):
    train_loader = DL(init.dataloader(init, 'train'), batch_size=init.batch_size,
                      shuffle=True, num_workers=init.num_workers, pin_memory=True, drop_last=False)

    valid_loader = DL(init.dataloader(init, 'valid'), batch_size=init.batch_size,
                      shuffle=False, num_workers=init.num_workers, pin_memory=True, drop_last=False)

    test_loader = DL(init.dataloader(init, 'test'), batch_size=init.batch_size,
                     shuffle=False, num_workers=init.num_workers, pin_memory=True, drop_last=False)
    return train_loader, valid_loader, test_loader


class ImageFolder(data.Dataset):
    def __init__(self, init: TrainInitializer, phase=None):
        tqdm.write(f"======================= Loading {init.dataset} {phase.title()} Datasets =======================")
        self.init = init
        self.phase = phase  # train, valid, test
        self.dataset = init.dataset
        self.task_type = init.task_type
        self.dataset_root, csv_path_dict = init.dataset_root, init.csv_path_dict
        self.transform = compose_transform(base_transform_define(init, phase))
        self.df = self.get_df(csv_path_dict[phase])
        self.imgs = self.df['name'].copy()

        # Fair
        if init.fair:
            self.gender = self.df['gender'].values
            self.age = self.df['age'].values
            self.race = self.df['race'].values

        # Labels
        if self.task_type == 'EXPR':
            self.labels, self.label_index = pd.factorize(self.df['expression'], sort=True)

        elif self.task_type == 'VA':
            self.valence = self.df['valence'].copy()
            self.arousal = self.df['arousal'].copy()
        elif self.task_type == 'AU':
            self.label_index = []
            for key in self.df.keys():
                if key not in ['name', 'age', 'race', 'gender', '99']:
                    self.label_index.append(key)
            self.label_index = sorted(self.label_index, key=lambda x: int(x[2:]))
            init.label_index = self.label_index
            self.labels = self.df[self.label_index].values

    def __getitem__(self, idx):
        # Images
        image_file_path = self.get_img_path(idx)
        img = self.transform(Image.open(image_file_path))

        # Labels
        age, gender, race = self.get_demographic_attribute(idx)
        label = self.get_label(idx)

        # Return
        self.dataset_root = f'{self.dataset_root}/' if self.dataset_root[-1] != '/' else self.dataset_root
        if self.init.fair:
            return img, label, age, gender, race, image_file_path.replace(self.dataset_root, '')
        else:
            return img, label, image_file_path.replace(self.dataset_root, '')

    def get_img_path(self, idx):
        image_file_path = os.path.join(self.dataset_root, self.imgs.iloc[idx])
        if self.dataset in ['RAF-AU']:
            image_file_path = f"{image_file_path.replace('.jpg', '_aligned.jpg')}"
        return image_file_path

    def get_label(self, idx):
        # Label
        if self.task_type in ['EXPR', 'AU']:
            label = self.labels[idx]
        elif self.task_type == 'VA':
            label = np.array([self.valence[idx], self.arousal[idx]])
        else:
            label = None
        return label

    def get_demographic_attribute(self, idx):
        # Fair
        if self.init.fair:
            age = self.age[idx]
            gender = self.gender[idx]
            race = self.race[idx]
        else:
            age, gender, race = None, None, None
        return age, gender, race

    def get_df(self, csv_path):
        if self.dataset == 'AffectNet-7':
            df = pd.read_csv(csv_path)  # (8327,5)
            df = df[df['expression'] != 'Contempt'].reset_index(drop=True)
        else:
            df = pd.read_csv(csv_path)  # (8327,5)
        # CSV Visualization
        tqdm.write(f'=> Dataset: {self.dataset}; Length: {len(df)}; CSV Path: {csv_path}')
        tqdm.write(f'=> DataHead:\n{df.head()}')
        return df

    def __len__(self):
        # return len(self.imgs)
        return 128
