# Copyright (c) 2024, Guanyu Hu
# All rights reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.


import torch
import torch.nn as nn
from torchvision.models import *
from models.iresgroup import iresgroup101
from torch.hub import load_state_dict_from_url
from torchvision.models._api import WeightsEnum


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


def get_gy_model(init):
    return GyModel(init)


class GyModel(nn.Module):
    def __init__(self, init, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init = init

        if init.model in ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']:
            if init.model == 'convnext_tiny':
                self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            elif init.model == 'convnext_small':
                self.model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
            elif init.model == 'convnext_base':
                self.model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
            elif init.model == 'convnext_large':
                self.model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)

            num_ftrs = self.model.classifier[2].in_features
            if init.task_type in ['AU', 'EXPR']:
                self.model.classifier[2] = nn.Linear(num_ftrs, init.num_class)
            elif init.task_type == 'VA':
                self.model.classifier[2] = nn.Identity()
                self.vhead = nn.Linear(num_ftrs, 1)
                self.ahead = nn.Linear(num_ftrs, 1)

        elif init.model == 'iresnet':
            self.model = iresgroup101(init, pretrained=True)
            num_ftrs = self.model.fc.in_features
            if init.task_type in ['AU', 'EXPR']:
                self.model.fc = nn.Linear(num_ftrs, init.num_class)
            elif init.task_type == 'VA':
                self.model.fc = nn.Identity()
                self.vhead = nn.Linear(num_ftrs, 1)
                self.ahead = nn.Linear(num_ftrs, 1)

        elif init.model in ['densenet121', 'densenet161', 'densenet201']:
            if init.model == 'densenet121':
                self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            elif init.model == 'densenet161':
                self.model = densenet161(weights=DenseNet161_Weights.DEFAULT)
            elif init.model == 'densenet201':
                self.model = densenet201(weights=DenseNet201_Weights.DEFAULT)

            num_ftrs = self.model.classifier.in_features
            if init.task_type in ['AU', 'EXPR']:
                self.model.classifier = nn.Linear(num_ftrs, init.num_class)
            elif init.task_type == 'VA':
                self.model.classifier = nn.Identity()
                self.vhead = nn.Linear(num_ftrs, 1)
                self.ahead = nn.Linear(num_ftrs, 1)

        elif init.model in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                            'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_s',
                            'efficientnet_v2_m', 'efficientnet_v2_l']:

            WeightsEnum.get_state_dict = get_state_dict

            if init.model == 'efficientnet_b0':
                self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            elif init.model == 'efficientnet_b1':
                self.model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
            elif init.model == 'efficientnet_b2':
                self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
            elif init.model == 'efficientnet_b6':
                self.model = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
            elif init.model == 'efficientnet_b7':
                self.model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
            elif init.model == 'efficientnet_v2_s':
                self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            elif init.model == 'efficientnet_v2_m':
                self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
            elif init.model == 'efficientnet_v2_l':
                self.model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)

            num_ftrs = self.model.classifier[1].in_features
            if init.task_type in ['AU', 'EXPR']:
                self.model.classifier = nn.Sequential(
                    nn.Dropout(p=0.5, inplace=True),
                    nn.Linear(num_ftrs, init.num_class))
            elif init.task_type == 'VA':
                self.model.classifier = nn.Sequential(
                    nn.Dropout(p=0.5, inplace=True),
                    nn.Identity())
                self.vhead = nn.Linear(num_ftrs, 1)
                self.ahead = nn.Linear(num_ftrs, 1)

        elif init.model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
                            'resnext101_32x8d', 'resnext101_64x4d']:
            if init.model == 'resnet18':
                self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            elif init.model == 'resnet34':
                self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
            elif init.model == 'resnet50':
                self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            elif init.model == 'resnet101':
                self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
            elif init.model == 'resnet152':
                self.model = resnet152(weights=ResNet152_Weights.DEFAULT)
            elif init.model == 'resnext50_32x4d':
                self.model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
            elif init.model == 'resnext101_32x8d':
                self.model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)
            elif init.model == 'resnext101_64x4d':
                self.model = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.DEFAULT)

            num_ftrs = self.model.fc.in_features
            if init.task_type in ['AU', 'EXPR']:
                self.model.fc = nn.Linear(num_ftrs, init.num_class)
            elif init.task_type == 'VA':
                self.model.fc = nn.Identity()
                self.vhead = nn.Linear(num_ftrs, 1)
                self.ahead = nn.Linear(num_ftrs, 1)

        elif init.model in ['swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b']:
            if init.model == 'swin_t':
                self.model = swin_t(weights=Swin_T_Weights.DEFAULT)
            elif init.model == 'swin_s':
                self.model = swin_s(weights=Swin_S_Weights.DEFAULT)
            elif init.model == 'swin_b':
                self.model = swin_b(weights=Swin_B_Weights.DEFAULT)
            elif init.model == 'swin_v2_t':
                self.model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
            elif init.model == 'swin_v2_s':
                self.model = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT)
            elif init.model == 'swin_v2_b':
                self.model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)

            num_ftrs = self.model.head.in_features
            if init.task_type in ['AU', 'EXPR']:
                self.model.head = nn.Linear(num_ftrs, init.num_class)
            elif init.task_type == 'VA':
                self.model.head = nn.Identity()
                self.vhead = nn.Linear(num_ftrs, 1)
                self.ahead = nn.Linear(num_ftrs, 1)

        elif init.model in ['vgg11', 'vgg16', 'vgg16_abaw', 'vgg19']:
            if init.model == 'vgg11':
                self.model = vgg11(weights=VGG11_Weights.DEFAULT)
            elif init.model == 'vgg16' or init.model == 'vgg16_abaw':
                self.model = vgg16(weights=VGG16_Weights.DEFAULT)
            elif init.model == 'vgg19':
                self.model = vgg19(weights=VGG19_Weights.DEFAULT)

            if init.task_type in ['AU', 'EXPR']:
                self.model.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, init.num_class),
                )
            elif init.task_type == 'VA':
                self.model.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=0.5),
                    nn.Identity(),
                )
                self.vhead = nn.Linear(4096, 1)
                self.ahead = nn.Linear(4096, 1)

        elif init.model in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']:
            if init.model == 'vit_b_16':
                self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            elif init.model == 'vit_b_32':
                self.model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
            elif init.model == 'vit_l_16':
                self.model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
            elif init.model == 'vit_l_32':
                self.model = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
            elif init.model == 'vit_h_14':
                self.model = vit_h_14(weights=ViT_H_14_Weights.DEFAULT)

            num_ftrs = self.model.heads[0].in_features
            if init.task_type in ['AU', 'EXPR']:
                self.model.heads = nn.Linear(num_ftrs, init.num_class)
            elif init.task_type == 'VA':
                self.model.heads = nn.Identity()
                self.vhead = nn.Linear(num_ftrs, 1)
                self.ahead = nn.Linear(num_ftrs, 1)
        else:
            raise Exception('!!!!!!! Wrong model type !!!!!!!')

    def forward(self, x):
        model_out = self.model(x)

        if self.init.task_type in ['AU', 'EXPR']:
            return model_out

        elif self.init.task_type == 'VA':
            v_out = self.vhead(model_out)
            a_out = self.ahead(model_out)
            return torch.cat((v_out, a_out), dim=1)
