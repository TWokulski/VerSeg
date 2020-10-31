import time
import torch
import sys
from .Data import CocoEvaluator, prepare_for_coco
from .Tools import CocoConversion


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    model.train()

    for i, (image, target) in enumerate(data_loader):
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        losses = model(image, target)
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if num_iters % args.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        if i >= iters - 1:
            break


def evaluate(model, data_loader, device, args, generate=True):
    dataset = data_loader
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(args.results, map_location="cpu")
    coco_evaluator.accumulate(results)

    temp = sys.stdout
    sys.stdout = CocoConversion()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp

    return output


