import random
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from model.to_cls import CLSTokenAggregator
from model.patch_77 import FeatureBlocker


def random_permutation(n):
    # 生成0到n的随机排列
    return list(np.random.permutation(n))


class sfm_model(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.mod_idx = base_model.mod_idx
        self.convert_output = None
        if self.mod_idx in [0, 1, 8]:
            def _(output):
                keys = list(base_model.features_we.keys())
                random_key = random.choice(keys) if len(keys) >0  else None
                if random_key is not None:
                    random_value = base_model.features_we[random_key]
                    f = torch.mean(random_value, dim=1).transpose(0, 1).contiguous()
                else:
                    f = None
                return output, f
            self.convert_output = _
            return

        if self.mod_idx in [3, 6, 2, 7, 9]:
            def _(output):
                keys = list(base_model.features_we.keys())
                random_key = random.choice(keys) if keys else None
                if random_key:
                    random_value = base_model.features_we[random_key]
                    f = torch.mean(random_value, dim=(2, 3)).transpose(0, 1).contiguous()
                else:
                    f = None
                return output, f
            self.convert_output = _
            return

        if self.mod_idx == 4:
            def _(output):
                f, v = output
                v = random.choice(v).transpose(0, 1)
                return f, v.contiguous()
            self.convert_output = _
            return

        assert NotImplementedError

    def forward(self, x):
        x = self.base_model(x)
        return self.convert_output(x)

