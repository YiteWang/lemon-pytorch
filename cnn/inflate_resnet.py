import datetime
import os
import time
import warnings

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from datasets import load_dataset, load_from_disk
from inflate_utils import inflate_resnet

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} Loss {metric_logger.loss.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )
        def hf_transform_test(examples):
            examples["image"] = [preprocessing(image.convert("RGB")) for image in examples["image"]]
            return examples
        dataset_huggingface = load_from_disk(args.data_path).with_format("torch")
        dataset_test = dataset_huggingface['validation']
        dataset_test.set_transform(hf_transform_test)
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_test, test_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    
    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    val_dir = os.path.join(args.data_path, "val")
    dataset_test, test_sampler = load_data(val_dir, args)

    # TODO: not hard-coding num_clases
    num_classes = 1000

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating source model:{}".format(args.src_model))
    src_model = torchvision.models.get_model(args.src_model, weights=args.weights, num_classes=num_classes)
    src_model.to(device)
    print("Total number of source model parameters: {}".format(sum(p.numel() for p in src_model.parameters())))

    print("Creating destination model:{}".format(args.dst_model))
    dst_model = torchvision.models.get_model(args.dst_model, weights=args.weights, num_classes=num_classes)
    dst_model.to(device)
    print("Total number of destination model parameters: {}".format(sum(p.numel() for p in dst_model.parameters())))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    assert args.resume is not None
    print("Load weights from {}".format(args.resume))
    checkpoint = torch.load(args.resume, map_location="cpu")
    src_model.load_state_dict(checkpoint["model"])
    args.start_epoch = checkpoint["epoch"] + 1

    # print("Start to evaluate source model")
    # evaluate(src_model, criterion, data_loader_test, device=device)

    print("Start to expand")
    inflate_resnet(src_model, dst_model, 'circular', 'circular', args.depth, args.method)

    print("Start to evaluate destination model")
    evaluate(dst_model, criterion, data_loader_test, device=device)

    checkpoint = {
        "model": dst_model.state_dict(),
        "args": args,
    }
    
    ckpt_path = os.path.join(args.output_dir, "{}_{}_{}_{}_{}.pth".format(
        args.method, args.src_model, args.dst_model, args.depth, args.start_epoch
    ))

    print("Save to: {}".format(ckpt_path))

    utils.save_on_master(checkpoint, ckpt_path)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch model expansion", add_help=add_help)

    parser.add_argument("--data-path", default="/dataset/imagenet/", type=str, help="dataset path")
    parser.add_argument("--src-model", default="resnet50", choices=["resnet18", "resnet50", "resnet101", "wide_resnet50_2", "wide_resnet101_2"], type=str, help="model name")
    parser.add_argument("--dst-model", default="wide_resnet101_2", choices=["resnet18", "resnet50", "resnet101", "wide_resnet50_2", "wide_resnet101_2"], type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=96, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--method", default="stack", type=str, choices=['lemon', 'fpi', 'aki'], help="expansion algorithm")
    parser.add_argument("--depth", default="stack", type=str, choices=['stack', 'interpolation'], help="how to stack depth in expansion")
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--output-dir", default="/mnt/bn/yitebn1/yite/data/cnn_inflate/resent50", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="/mnt/bn/yitebn1/yite/data/cnn_pretrain/resnet50/T1/checkpoint.pth", type=str, help="path of source model ckpt")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
