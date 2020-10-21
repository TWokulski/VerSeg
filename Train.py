import bisect
import glob
import os
import re
import time
import torch
import Mask_RCNN as algorithm
from Config import *


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print("\ndevice: {}".format(device))

    train_set = algorithm.datasets(dataset_dir, "train2017", train=True)
    indices = torch.randperm(len(train_set)).tolist()
    train_set = torch.utils.data.Subset(train_set, indices)

    val_set = algorithm.datasets(dataset_dir, "val2017", train=True)
    model = algorithm.resnet50_for_mask_rcnn(True).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=momentum, weight_decay=decay)
    decrease = lambda x: learning_rate_lambda ** bisect.bisect(learning_rate_steps, x)

    starting_epoch = 0
    prefix, ext = os.path.splitext(ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device)  # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(starting_epoch, number_of_epochs))

    for epoch in range(starting_epoch, number_of_epochs):
        print("\nepoch: {}".format(epoch + 1))

        training_epoch_time = time.time()
        lr_epoch = decrease(epoch) * learning_rate
        print("lr_epoch: {:.4f}, factor: {:.4f}".format(lr_epoch, decrease(epoch)))
        iter_train = algorithm.train_one_epoch(model, optimizer, train_set, device, epoch)
        training_epoch_time = time.time() - training_epoch_time

        validation_epoch_time = time.time()
        eval_output, iter_eval = algorithm.evaluate(model, val_set, device)
        validation_epoch_time = time.time() - validation_epoch_time

        trained_epoch = epoch + 1
        print("training: {:.2f} s, evaluation: {:.2f} s".format(training_epoch_time, validation_epoch_time))
        algorithm.collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        print(eval_output.get_AP())

        algorithm.save_ckpt(model, optimizer, trained_epoch, eval_info=str(eval_output))

    print("\ntotal time of this training: {:.2f} s".format(time.time() - since))
    if starting_epoch < number_of_epochs:
        print("already trained: {} epochs\n".format(trained_epoch))


if __name__ == "__main__":
    main()
