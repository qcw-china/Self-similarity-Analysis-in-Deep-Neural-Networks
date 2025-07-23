# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import math
import os
import time
from collections import defaultdict, deque
import datetime

import numpy as np
import torch
import torch.distributed as dist

from utils import SmoothedValue
from utils import is_dist_avail_and_initialized


def std_mean(x):
    assert isinstance(x, torch.Tensor)
    std, mean = torch.std_mean(x)
    return std.item(), mean.item()


class SmoothedTensor(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "global_mean_std:{std:.4f} {mean:.4f}"
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        self.count = int(t[0])
        # self.total:torch.Tensor
        dist.barrier()
        dist.all_reduce(self.total)

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        std, mean = std_mean(self.global_avg)
        return self.fmt.format(
            std=std,
            mean=mean,
        )


class MetricLoggerTensor(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedTensor)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(v, torch.Tensor)
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def sample_index(hid_dim, rate, layer):
    i = np.random.choice(range(hid_dim), int(rate * hid_dim))
    i = [torch.from_numpy(i)]
    for j in range(layer - 1):
        i.append(i[0] + hid_dim * (j + 1))
    i = torch.cat(i, dim=0)
    i = i.long()
    return i


def find_nearest_divisible(n, target):
    for i in range(target, 0, -1):
        if n % i == 0:
            return i
    return 1


def compress_tensor(tensor, compress_ratio):
    n, d = tensor.shape
    target_new_n = int(n * compress_ratio)
    new_n = find_nearest_divisible(n, target_new_n)
    block_size = n // new_n
    compressed_tensor = tensor.view(new_n, block_size, -1).mean(dim=1)
    return compressed_tensor


"""
    限制sfm损失在某个范围内
"""


def control_sfm_loss(sfm_loss, A, B):
    # 1控制rate比例，也即S
    # 2控制分析维数，也即K
    return A*sfm_loss[0] + B * sfm_loss[1]


def get_cos(_min, _max, total_epochs, current_epoch):
    if current_epoch<0:
        return _max
    if current_epoch > total_epochs:
        return _min
    cos_inner = math.pi * current_epoch / total_epochs
    lr = _min + 0.5 * (_max - _min) * (1 + math.cos(cos_inner))
    return lr

def smooth_transition(min_val, max_val, epochs):
    if epochs <= 1:
        return [max_val]

    transition = []
    delta = max_val - min_val

    curitledsa= 100
    for t in range(epochs):
        x = t / (epochs - 1)
        y = np.log(curitledsa * x+1) / np.log(1 + curitledsa) # 核心曲率控制公式
        current = min_val + delta * y
        transition.append(round(current, 4))

    return transition
