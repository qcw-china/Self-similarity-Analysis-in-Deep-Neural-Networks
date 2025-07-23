# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import random
import sys

import timm
import torch
import torch.nn as nn
from functools import partial

from timm.layers import LayerNorm2d, ClassifierHead
from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from torch.nn import Dropout, Linear

"""
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        img_size=args.input_size
    )
    pretrained, num_classes, img_size
    use_sim_loss
"""

# 所有实例版本的模型的名字
__all__ = [
    'vit_small_', 'vit_base_', 'resmlp_12_224_we', 'pvt_v2', 'mlp_512', 'poolformer', 'resnet_34_timm',
    'resnet_50_timm', 'mxier', 'vgg'
]


@register_model
def vit_small_(pretrained=False, num_classes=10, img_size=224, output_hidden_states=False,
               pretrained_cfg=None, pretrained_cfg_overlay=None, args=None,cache_dir=None):
    model = timm.create_model(
        'vit_small_patch16_224',
        pretrained=False,
        num_classes=num_classes,
    )
    model.features_we = {}

    # 定义钩子函数
    def get_feature(name):
        def hook(_, input, output):
            model.features_we[name] = output.clone()

        return hook

    if args.use_sfm_loss or args.only_record_sfm_rate:
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 2 and 'blocks' in nl[0]:
                m.register_forward_hook(get_feature(n))

    if pretrained:
        assert NotImplementedError
    model.mod_idx = 0
    return model


@register_model
def vit_base_(pretrained=False, num_classes=10, img_size=224, output_hidden_states=False,
              pretrained_cfg=None, pretrained_cfg_overlay=None, args=None,cache_dir=None):
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=num_classes,
    )
    model.features_we = {}

    # 定义钩子函数
    def get_feature(name):
        def hook(_, input, output):
            model.features_we[name] = output.clone()

        return hook

    if args.use_sfm_loss or args.only_record_sfm_rate:
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 2 and 'blocks' in nl[0]:
                m.register_forward_hook(get_feature(n))

    if pretrained:
        assert NotImplementedError
    model.mod_idx = 0
    return model


@register_model
def resmlp_12_224_we(pretrained=False, num_classes=10, img_size=224, output_hidden_states=False, pretrained_cfg=None,
                     pretrained_cfg_overlay=None, args=None,cache_dir=None):
    model = timm.create_model(
        'resmlp_12_224.fb_in1k',
        pretrained=False,
        num_classes=num_classes,
        # global_pool='',  # 禁用全局平均池化
    )
    model.features_we = {}

    def gen_hook(layer):
        def hook(mod, input, output):
            model.features_we[layer] = output

        return hook

    if args.use_sfm_loss or args.only_record_sfm_rate:
        for n, m in model.named_modules():
            g = n.split('.')
            if len(g) == 2 and g[0] == 'blocks':
                m.register_forward_hook(gen_hook(str(int(g[1]) + 1)))

        model.stem.register_forward_hook(gen_hook(0))

    model.mod_idx = 1
    return model


@register_model
def pvt_v2(pretrained=False, num_classes=10, img_size=224, output_hidden_states=False,
           pretrained_cfg=None, pretrained_cfg_overlay=None, args=None,cache_dir=None):
    model = timm.create_model(
        'pvt_v2_b3',
        pretrained=False,
        num_classes=num_classes
    )

    model.features_we = {}

    # 定义钩子函数
    def get_feature(name):
        def hook(_, input, output):
            model.features_we[name] = output.clone()

        return hook

    if args.use_sfm_loss or args.only_record_sfm_rate:
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 4 and 'stages' in nl[0] and 'blocks' in nl[2]:
                m.register_forward_hook(get_feature(n))

    model.mod_idx = 3
    return model


import torch.nn.init as init
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, layer, num_classes=10):
        super(MLP, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)  # 假设输入为224x224x3的图像
        self.out_linear = nn.Linear(hidden_dim, num_classes)

        self.hid_forward = torch.nn.ModuleList()
        for _ in range(layer):
            self.hid_forward.append(
                timm.layers.Mlp(in_features=hidden_dim, hidden_features=4 * hidden_dim, norm_layer=None))
        self.layer = layer
        self.dropout = nn.Dropout(p=0.3)  # Dropout层，适当降低过拟合Risk

    def forward(self, x):
        # 展平输入图像
        x = x.view(x.size(0), -1)  # Flatten layer: (batch_size, 224*224*3)
        x = self.in_linear(x)
        vec = [x]
        for i in range(self.layer):
            x = self.hid_forward[i](x) + x
            vec.append(x)
            # x = x1 + x
        x = self.dropout(x)  # 应用dropout
        x = self.out_linear(x)
        return x, vec


@register_model
def mlp_512(pretrained=False, num_classes=10, img_size=224, output_hidden_states=False,
            pretrained_cfg=None, pretrained_cfg_overlay=None, args=None,cache_dir=None):
    model = MLP(in_dim=img_size ** 2 * 3, hidden_dim=512, layer=5, num_classes=num_classes)
    if pretrained:
        # Load pretrained weights if desired (place pre-trained weights loading logic here)
        pass
    model.mod_idx = 4
    return model


@register_model
def poolformer(pretrained=False, num_classes=10, img_size=224, output_hidden_states=False,
               pretrained_cfg=None, pretrained_cfg_overlay=None, args=None,cache_dir=None):
    model = timm.create_model(
        "poolformer_s12",  # 模型名称（支持 s12/s24/s36/m36 等变体）
        pretrained=False,  # 加载预训练权重
        num_classes=num_classes,  # 输出类别数（默认ImageNet的1000类）
    )
    model.features_we = {}

    # 定义钩子函数
    def get_feature(name):
        def hook(_, input, output):
            model.features_we[name] = output.clone()

        return hook

    if args.use_sfm_loss or args.only_record_sfm_rate:
        for i in range(0, 4):
            for idx, block in enumerate(model.stages[i].blocks):  # 假设模型的主要模块是 `blocks`
                block.register_forward_hook(get_feature(f'stages.{i}.blocks.{idx}'))

    model.mod_idx = 6
    return model


@register_model
def resnet_50_timm(pretrained=False, num_classes=10, img_size=224, output_hidden_states=False,
                   pretrained_cfg=None, pretrained_cfg_overlay=None, args=None,cache_dir=None):
    model = timm.create_model(
        "resnet50",  # 模型名称（支持 s12/s24/s36/m36 等变体）
        pretrained=False,  # 加载预训练权重
        num_classes=num_classes,  # 输出类别数（默认ImageNet的1000类）
    )
    model.features_we = {}

    # 定义钩子函数
    def get_feature(name):
        def hook(_, input, output):
            model.features_we[name] = output.clone()

        return hook

    if args.use_sfm_loss or args.only_record_sfm_rate:
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 2 and 'layer' in nl[0]:
                m.register_forward_hook(get_feature(n))

    if pretrained:
        assert NotImplementedError

    model.mod_idx = 2
    return model


@register_model
def resnet_34_timm(pretrained=False, num_classes=10, img_size=224, output_hidden_states=False,
                   pretrained_cfg=None, pretrained_cfg_overlay=None, args=None,cache_dir=None):
    model = timm.create_model(
        "resnet34",  # 模型名称（支持 s12/s24/s36/m36 等变体）
        pretrained=False,  # 加载预训练权重
        num_classes=num_classes,  # 输出类别数（默认ImageNet的1000类）
    )
    model.features_we = {}

    # 定义钩子函数
    def get_feature(name):
        def hook(_, input, output):
            model.features_we[name] = output.clone()

        return hook

    if args.use_sfm_loss or args.only_record_sfm_rate:
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 2 and 'layer' in nl[0]:
                m.register_forward_hook(get_feature(n))

    if pretrained:
        assert NotImplementedError

    model.mod_idx = 7
    return model


@register_model
def mxier(pretrained=False, num_classes=10, img_size=224, output_hidden_states=False,
          pretrained_cfg=None, pretrained_cfg_overlay=None, args=None,cache_dir=None):
    model = timm.create_model(
        "mixer_b16_224",  # 模型名称（支持 s12/s24/s36/m36 等变体）
        pretrained=False,  # 加载预训练权重
        num_classes=num_classes,  # 输出类别数（默认ImageNet的1000类）
    )

    model.features_we = {}

    # 定义钩子函数
    def get_feature(name):
        def hook(_, input, output):
            model.features_we[name] = output.clone()

        return hook

    if args.use_sfm_loss or args.only_record_sfm_rate:
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 2 and 'blocks' in nl[0]:
                m.register_forward_hook(get_feature(n))

    if pretrained:
        assert NotImplementedError

    model.mod_idx = 8
    return model


@register_model
def vgg(pretrained=False, num_classes=10, img_size=224, output_hidden_states=False,
        pretrained_cfg=None, pretrained_cfg_overlay=None, args=None,cache_dir=None):
    model = timm.create_model(
        "vgg11",  # 模型名称（支持 s12/s24/s36/m36 等变体）
        pretrained=False,  # 加载预训练权重
        num_classes=num_classes,  # 输出类别数（默认ImageNet的1000类）
    )

    model.features_we = {}

    # 定义钩子函数
    def get_feature(name):
        def hook(_, input, output):
            model.features_we[name] = output.clone()

        return hook

    if args.use_sfm_loss or args.only_record_sfm_rate:
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 2 and 'features' in nl[0]:
                m.register_forward_hook(get_feature(n))

    if pretrained:
        assert NotImplementedError

    model.mod_idx = 9
    return model
