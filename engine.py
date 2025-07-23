# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from sfm.sfm_loss_d_v2 import compute_mse_sim_loss
from sfm.sfm_utils import control_sfm_loss, get_cos
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.cosub:
            samples = torch.cat((samples, samples), dim=0)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

            VFS = outputs[1]
            outputs = outputs[0]

            if not args.cosub:
                loss1 = criterion(outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0] // 2, dim=0)
                loss1 = 0.25 * criterion(outputs[0], targets)
                loss1 = loss1 + 0.25 * criterion(outputs[1], targets)
                loss1 = loss1 + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss1 = loss1 + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())


            if args.use_sfm_loss or args.only_record_sfm_rate:
                loss2, TZ, TV, now_rate = compute_mse_sim_loss(VFS, target_rate=args.target_sfm_rate)

            z = 1
            if args.use_sfm_loss:
                if getattr(args, 'old_ss', None):
                    z = min(abs((now_rate - args.old_ss) / args.old_ss),
                            args.Change_amplitude_limit)
                    z = (1 - z / args.Change_amplitude_limit)
                else:
                    z=0;

            if args.use_sfm_loss:
                loss = loss1 + z * args.sfm_weight * loss2
            elif args.only_record_sfm_rate:
                loss = loss1 + 0. * loss2
            else:
                loss = loss1

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss1=loss1.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if args.use_sfm_loss or args.only_record_sfm_rate:
            metric_logger.update(loss2=loss2.item())
            metric_logger.update(TZ=TZ)
            metric_logger.update(TV=TV)
            metric_logger.update(now_rate=now_rate)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if args.use_sfm_loss or args.only_record_sfm_rate:
        if not hasattr(args, 'old_epoch'):
            args.old_epoch = 0
            args.old_ss = metric_logger.now_rate.global_avg
        elif epoch - args.old_epoch >= args.sample_epoch_fre:
            args.old_epoch = epoch
            args.old_ss = metric_logger.now_rate.global_avg

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)

            output = output[0]
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
