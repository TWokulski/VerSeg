import bisect
import glob
import os
import re
import time
import torch
import Mask_RCNN as algorithm
from Config import Configuration


def main():

    cfg = Configuration()
    parameters = {
        "seed": cfg.seed,
        "number_of_epochs": cfg.number_of_epochs,
        "number_of_classes": cfg.number_of_classes,
        "number_of_iterations": cfg.number_of_iterations,
        "momentum": cfg.momentum,
        "decay": cfg.decay,
        "learning_rate": cfg.learning_rate,
        "learning_steps": cfg.learning_rate_steps,
        "device": cfg.device,
        "dataset_dir": cfg.dataset_dir,
        "publishing_losses_frequency": 100,
        "checkpoint_path": cfg.ckpt_path,
        "learning_rate_lambda": cfg.learning_rate_lambda,
        "model_path": cfg.model_path,
        "iterations_to_warmup": 800,
        "result_path": cfg.result_path
    }

    device = torch.device(parameters['device'])
    best_model_by_maskF1 = 0

    train_set = algorithm.COCODataset(parameters['dataset_dir'], "Train", train=True)
    indices = torch.randperm(len(train_set)).tolist()
    train_set = torch.utils.data.Subset(train_set, indices)

    val_set = algorithm.COCODataset(parameters['dataset_dir'], "Validation", train=True)
    model = algorithm.resnet50_for_mask_rcnn(True, parameters['number_of_classes']).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=parameters['learning_rate'],
        momentum=parameters['momentum'],
        weight_decay=parameters['decay'])

    decrease = lambda x: parameters['learning_rate_lambda'] ** bisect.bisect(
        parameters['learning_steps'], x)

    starting_epoch = 0
    prefix, ext = os.path.splitext(parameters['checkpoint_path'])
    checkpoints = glob.glob(prefix + "-*" + ext)
    checkpoints.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if checkpoints:
        checkpoint = torch.load(checkpoints[-1], map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(starting_epoch, parameters['number_of_epochs']))

    for epoch in range(starting_epoch, parameters['number_of_epochs']):
        print("\nepoch: {}".format(epoch + 1))

        training_epoch_time = time.time()
        parameters['learning_epoch'] = decrease(epoch) * parameters['learning_rate']

        algorithm.train_epoch(model, optimizer, train_set, device, epoch, parameters)
        training_epoch_time = time.time() - training_epoch_time
        print('training_epoch_time: ', training_epoch_time)

        validation_epoch_time = time.time()
        eval_output = algorithm.evaluate(model, val_set, device, parameters)
        validation_epoch_time = time.time() - validation_epoch_time
        print('validation_epoch_time: ', validation_epoch_time)

        trained_epoch = epoch + 1
        maskAP = eval_output.get_AP()
        maskAR = eval_output.get_AR()
        maskF1 = eval_output.get_AF1()
        print('AP: ', maskAP)
        print('AR: ', maskAR)
        print('F1: ', maskF1)
        if maskF1['mask F1Score'] > best_model_by_maskF1:
            best_model_by_maskF1 = maskF1['mask F1Score']
            algorithm.save_best(model, optimizer, trained_epoch,
                                parameters['model_path'], eval_info=str(eval_output))

        algorithm.save_checkpoint(model, optimizer, trained_epoch,
                                  parameters['checkpoint_path'], eval_info=str(eval_output))

        prefix, ext = os.path.splitext(parameters['checkpoint_path'])
        checkpoints = glob.glob(prefix + "-*" + ext)
        checkpoints.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 3
        if len(checkpoints) > n:
            for i in range(len(checkpoints) - n):
                os.remove("{}".format(checkpoints[i]))

    total_training_time = time.time() - since
    print('Total time: ', total_training_time)


if __name__ == "__main__":
    main()
