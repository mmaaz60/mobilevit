#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import multiprocessing
import torch
from utils import logger
from options.opts import get_training_arguments
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master, distributed_init
from cvnets import get_model, EMA
from loss_fn import build_loss_fn
from optim import build_optimizer
from optim.scheduler import build_scheduler
from data import create_train_val_loader
from utils.checkpoint_utils import load_checkpoint, load_model_state
from engine import Trainer
import math
from torch.cuda.amp import GradScaler
from common import DEFAULT_EPOCHS, DEFAULT_ITERATIONS, DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_EPOCHS
from ptflops import get_model_complexity_info
from copy import deepcopy
from fvcore.nn import FlopCountAnalysis
import time


def main(opts, **kwargs):
    num_gpus = getattr(opts, "dev.num_gpus", 0) # defaults are for CPU
    dev_id = getattr(opts, "dev.device_id", torch.device('cpu'))
    device = getattr(opts, "dev.device", torch.device('cpu'))
    is_distributed = getattr(opts, "ddp.use_distributed", False)

    is_master_node = is_master(opts)

    # set-up data loaders
    train_loader, val_loader, train_sampler = create_train_val_loader(opts)

    # compute max iterations based on max epochs
    # Useful in doing polynomial decay
    is_iteration_based = getattr(opts, "scheduler.is_iteration_based", False)
    if is_iteration_based:
        max_iter = getattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
        if max_iter is None or max_iter <= 0:
            logger.log('Setting max. iterations to {}'.format(DEFAULT_ITERATIONS))
            setattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
            max_iter = DEFAULT_ITERATIONS
        setattr(opts, "scheduler.max_epochs", DEFAULT_MAX_EPOCHS)
        if is_master_node:
            logger.log('Max. iteration for training: {}'.format(max_iter))
    else:
        max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        if max_epochs is None or max_epochs <= 0:
            logger.log('Setting max. epochs to {}'.format(DEFAULT_EPOCHS))
            setattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        setattr(opts, "scheduler.max_iterations", DEFAULT_MAX_ITERATIONS)
        max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        if is_master_node:
            logger.log('Max. epochs for training: {}'.format(max_epochs))
    # set-up the model
    model = get_model(opts)

    if num_gpus == 0:
        logger.error('Need atleast 1 GPU for training. Got {} GPUs'.format(num_gpus))
    elif num_gpus == 1:
        model = model.to(device=device)
    elif is_distributed:
        model = model.to(device=device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
        if is_master_node:
            logger.log('Using DistributedDataParallel for training')
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device=device)
        if is_master_node:
            logger.log('Using DataParallel for training')
    # Stats only
    stats_only = getattr(opts, "common.stats_only", 0)
    if stats_only:
        width = getattr(opts, "sampler.vbs.crop_size_width", 256)
        height = getattr(opts, "sampler.vbs.crop_size_height", 256)
        # Calculate Model FLOPS
        macs, params = get_model_complexity_info(model, (3, width, height), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print(f"Params & Flops using ptflops:")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # Print layer wise params
        from prettytable import PrettyTable

        def count_parameters(model):
            table = PrettyTable(["Modules", "Parameters"])
            total_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad: continue
                params = parameter.numel()
                table.add_row([name, params])
                total_params += params
            print(table)
            return total_params

        total_params = count_parameters(model)

        # fvcore
        input_res = (3, width, height)
        input = torch.ones(()).new_empty((1, *input_res), dtype=next(model.parameters()).dtype,
                                         device=next(model.parameters()).device)
        flops = FlopCountAnalysis(model, input)
        model_flops = flops.total()
        print(f"Total Trainable Params: {round(total_params * 1e-6, 2)} M")
        print(f"Flops using fvcore: {round(model_flops * 1e-6, 2)} M")

        # Using Model summary
        from model_summary import get_model_activation, get_model_flops
        input_dim = (3, 256, 256)  # set the input dimension

        activations, num_conv2d = get_model_activation(model, input_dim)
        print('{:>16s} : {:<.4f} [M]'.format('#Activations', activations / 10 ** 6))
        print('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))

        flops = get_model_flops(model, input_dim, False)
        print('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops / 10 ** 9))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters / 10 ** 6))

        # Measure the FPS
        input_res = (3, width, height)
        batch = torch.ones(()).new_empty((1, *input_res),
                                         dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
        model_fps = deepcopy(model)
        model_fps.eval()
        inference_times = []
        with torch.no_grad():
            # A few times pass the inputs to the network - warm-up
            for i in range(10):
                preds = model(batch)
            for i in range(1000):
                start = time.time()
                _ = model_fps(batch)
                end = time.time()
                inference_times.append(end - start)
            print(f"FPS @ BS=1: {round(1 / (sum(inference_times) / len(inference_times)), 2)}")
            # Measure the FPS @ BS=256
            inference_times = []
            batch_size = 256
            input_res = (3, width, height)
            inputs = torch.ones(()).new_empty((batch_size, *input_res),
                                              dtype=next(model.parameters()).dtype,
                                              device=next(model.parameters()).device)
            # A few times pass the inputs to the network - warm-up
            for i in range(2):
                preds = model(inputs)
            # Perform the actual benchmarking following DeiT paper
            end = time.time()
            for i in range(30):
                # Run the model forward pass
                preds = model(inputs)
                inference_times.append(time.time() - end)
                end = time.time()
        print(f"Total passes to network: {30}")
        print(f"Batch size: {batch_size}")
        print(f"Total network time: {sum(inference_times)} sec")
        print(f"Throughput: {30 * batch_size / sum(inference_times)} FPS")

        return 0

    # setup criteria
    criteria = build_loss_fn(opts)
    criteria = criteria.to(device=device)

    # create the optimizer
    optimizer = build_optimizer(model, opts=opts)

    # create the gradient scalar
    gradient_scalar = GradScaler(
        enabled=getattr(opts, "common.mixed_precision", False)
    )

    # LR scheduler
    scheduler = build_scheduler(opts=opts)

    model_ema = None
    use_ema = getattr(opts, "ema.enable", False)

    if use_ema:
        ema_momentum = getattr(opts, "ema.momentum", 0.0001)
        model_ema = EMA(
            model=model,
            ema_momentum=ema_momentum,
            device=device
        )
        if is_master_node:
            logger.log('Using EMA')

    best_metric = 0.0 if getattr(opts, "stats.checkpoint_metric_max", False) else math.inf

    start_epoch = 0
    start_iteration = 0
    resume_loc = getattr(opts, "common.resume", None)
    finetune_loc = getattr(opts, "common.finetune", None)
    auto_resume = getattr(opts, "common.auto_resume", False)
    if resume_loc is not None or auto_resume:
        model, optimizer, gradient_scalar, start_epoch, start_iteration, best_metric, model_ema = load_checkpoint(
            opts=opts,
            model=model,
            optimizer=optimizer,
            model_ema=model_ema,
            gradient_scalar=gradient_scalar
        )
    elif finetune_loc is not None:
        model, model_ema = load_model_state(opts=opts, model=model, model_ema=model_ema)
        if is_master_node:
            logger.log('Finetuning model from checkpoint {}'.format(finetune_loc))

    training_engine = Trainer(opts=opts,
                              model=model,
                              validation_loader=val_loader,
                              training_loader=train_loader,
                              optimizer=optimizer,
                              criterion=criteria,
                              scheduler=scheduler,
                              start_epoch=start_epoch,
                              start_iteration=start_iteration,
                              best_metric=best_metric,
                              model_ema=model_ema,
                              gradient_scalar=gradient_scalar
                              )

    training_engine.run(train_sampler=train_sampler)


def distributed_worker(i, main, opts, kwargs):
    setattr(opts, "dev.device_id", i)
    if torch.cuda.is_available():
        torch.cuda.set_device(i)

    ddp_rank = getattr(opts, "ddp.rank", None)
    if ddp_rank is None:  # torch.multiprocessing.spawn
        ddp_rank = kwargs.get('start_rank', 0) + i
        setattr(opts, "ddp.rank", ddp_rank)

    node_rank = distributed_init(opts)
    setattr(opts, "ddp.rank", node_rank)
    main(opts, **kwargs)


def main_worker(**kwargs):
    opts = get_training_arguments()
    print(opts)
    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank", 0)
    if node_rank < 0:
        logger.error('--rank should be >=0. Got {}'.format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = '{}/{}'.format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus", 1)
    world_size = getattr(opts, "ddp.world_size", -1)
    use_distributed = getattr(opts, "ddp.enable", False)
    if num_gpus <= 1:
        use_distributed = False
    setattr(opts, "ddp.use_distributed", use_distributed)

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = multiprocessing.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)

    norm_name = getattr(opts, "model.normalization.name", "batch_norm")
    if use_distributed:
        if world_size == -1:
            logger.log("Setting --ddp.world-size the same as the number of available gpus")
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)
        elif world_size != num_gpus:
            logger.log("--ddp.world-size does not match num. available GPUs. Got {} !={}".format(world_size, num_gpus))
            logger.log("Setting --ddp.world-size=num_gpus")
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)

        if dataset_workers == -1 or dataset_workers is None:
            setattr(opts, "dataset.workers", n_cpus // world_size)

        start_rank = getattr(opts, "ddp.rank", 0)
        setattr(opts, "ddp.rank", None)
        kwargs['start_rank'] = start_rank
        torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, opts, kwargs),
            nprocs=num_gpus,
        )
    else:
        if dataset_workers == -1:
            setattr(opts, "dataset.workers", n_cpus)

        if norm_name in ["sync_batch_norm", "sbn"]:
            setattr(opts, "model.normalization.name", "batch_norm")

        # adjust the batch size
        train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
        val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
        setattr(opts, "dataset.train_batch_size0", train_bsize)
        setattr(opts, "dataset.val_batch_size0", val_bsize)
        setattr(opts, "dev.device_id", None)
        main(opts=opts, **kwargs)


if __name__ == "__main__":
    #multiprocessing.set_start_method('spawn', force=True)

    main_worker()