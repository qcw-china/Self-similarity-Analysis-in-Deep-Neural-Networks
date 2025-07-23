"""
    实现一个统计
"""
import sys

import numpy as np
import torch
import torch.nn as nn

from sfm import sfm_utils

"""
    统计在数据集上平均的距离矩阵，统计所有的神经元
"""


@torch.no_grad()
def stat_whole_adj_on_dataset(model: nn.Module, dataloader, device):
    metric_logger = sfm_utils.MetricLoggerTensor(delimiter="  ")
    header = 'Test_for_sfm:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(dataloader, 50, header):
        images = images.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            assert isinstance(output, tuple) and len(output) > 1
            vfs = output[1]

        batch_size = images.shape[0]
        if batch_size == dataloader.batch_size:
            metric_logger.meters['vfs'].update(vfs, n=1)
        break
        
    metric_logger.synchronize_between_processes()
    print('* final:'+str(metric_logger.meters['vfs']))

    return metric_logger.meters['vfs'].global_avg
