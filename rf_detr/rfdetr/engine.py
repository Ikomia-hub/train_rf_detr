# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable

import torch

import train_rf_detr.rf_detr.rfdetr.util.misc as utils
from train_rf_detr.rf_detr.rfdetr.datasets.coco_eval import CocoEvaluator

try:
    from torch.amp import autocast, GradScaler
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    DEPRECATED_AMP = True
from typing import DefaultDict, List, Callable
from train_rf_detr.rf_detr.rfdetr.util.misc import NestedTensor

# Attempt to import mlflow
try:
    import mlflow
    assert hasattr(mlflow, '__version__')
except (ImportError, AssertionError):
    mlflow = None


def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {'enabled': args.amp, 'dtype': torch.bfloat16}
    else:
        return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", utils.SmoothedValue(
        window_size=1, fmt="{value:.2f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch

    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", batch_size * utils.get_world_size())

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler('cuda', enabled=args.amp)

    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps

    # Initialize MLflow run if MLflow is available and no run is active
    if mlflow and mlflow.active_run() is None:
        mlflow.set_experiment("LW_DETR_Experiments")
        mlflow.start_run(run_name="Train_Training")

    print("LENGTH OF DATA LOADER:", len(data_loader))
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + data_iter_step
        callback_dict = {
            "step": it,
            "model": model,
            "epoch": epoch,
        }
        if callbacks is not None and "on_train_batch_start" in callbacks:
            for callback in callbacks["on_train_batch_start"]:
                callback(callback_dict)
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(
                    schedules["dp"][it], vit_encoder_num_layers)
            else:
                model.update_drop_path(
                    schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        for i in range(args.grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples_tensors = samples.tensors[start_idx:final_idx]
            new_samples = NestedTensor(
                new_samples_tensors, samples.mask[start_idx:final_idx])
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets[start_idx:final_idx]]

            with autocast(**get_autocast_args(args)):
                outputs = model(new_samples, new_targets)
                loss_dict = criterion(outputs, new_targets)
                weight_dict = criterion.weight_dict
                losses = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )

            scaler.scale(losses).backward()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k:  v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Log training metrics to MLflow every iteration (only on main process)
        if mlflow and (not args.distributed or args.rank == 0):
            mlflow.log_metric("train_loss", loss_value, step=it)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=it)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Log aggregated epoch metrics to MLflow (only on main process)
    if mlflow and (not args.distributed or args.rank == 0):
        for key, meter in metric_logger.meters.items():
            mlflow.log_metric(f"epoch_{key}", meter.global_avg, step=epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None, epoch=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("class_error", utils.SmoothedValue(
        window_size=1, fmt="{value:.2f}"))
    header = "Test:"

    # Determine which IoU types to evaluate (e.g., 'bbox' and/or 'segm')
    iou_types = tuple(k for k in ("segm", "bbox")
                      if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # Initialize MLflow run for evaluation if not already active
    if mlflow and mlflow.active_run() is None:
        mlflow.start_run(run_name="Evaluation")

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        with autocast(**get_autocast_args(args)):
            outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][sub_key].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        res = {target["image_id"].item(): output for target,
               output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
    if coco_evaluator is not None and mlflow:
        stats_array = coco_evaluator.coco_eval['bbox'].stats
        if len(stats_array) >= 12:
            # Log precision (AP) metrics
            mlflow.log_metric("bbox_AP_50_95_all", stats_array[0], step=epoch)
            mlflow.log_metric("bbox_AP_50_all", stats_array[1], step=epoch)
            mlflow.log_metric("bbox_AP_75_all", stats_array[2], step=epoch)
            mlflow.log_metric("bbox_AP_small", stats_array[3], step=epoch)
            mlflow.log_metric("bbox_AP_medium", stats_array[4], step=epoch)
            mlflow.log_metric("bbox_AP_large", stats_array[5], step=epoch)
            # Log recall (AR) metrics
            mlflow.log_metric("bbox_AR_max1_all", stats_array[6], step=epoch)
            mlflow.log_metric("bbox_AR_max10_all", stats_array[7], step=epoch)
            mlflow.log_metric("bbox_AR_max100_all", stats_array[8], step=epoch)
            mlflow.log_metric("bbox_AR_small", stats_array[9], step=epoch)
            mlflow.log_metric("bbox_AR_medium", stats_array[10], step=epoch)
            mlflow.log_metric("bbox_AR_large", stats_array[11], step=epoch)
        else:
            print(
                "Warning: COCO evaluation stats are empty or incomplete, skipping metric logging.")
            mlflow.log_metric("bbox_stats_incomplete", 1, step=epoch)
        # Pass the epoch number to the summarize method
        coco_evaluator.summarize(epoch=epoch)

    # Create stats dictionary from metric logger meters
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # Add COCO bbox evaluation stats if available
    if coco_evaluator is not None:
        bbox_stats = coco_evaluator.coco_eval['bbox'].stats
        if len(bbox_stats) >= 12:
            stats["bbox"] = bbox_stats
            stats["coco_eval_bbox"] = bbox_stats

    # Log the evaluation metrics from the metric logger to MLflow as well
    if mlflow and (not args.distributed or args.rank == 0):
        for key, meter in metric_logger.meters.items():
            mlflow.log_metric(f"eval_{key}", meter.global_avg, step=epoch)
    return stats, coco_evaluator
