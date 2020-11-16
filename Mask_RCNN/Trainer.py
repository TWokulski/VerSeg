import torch
import sys
from .Data import CocoEvaluator, prepare_for_coco
from .Tools import CocoConversion


def train_epoch(model, optimizer, data_loader, device, epoch, parameters):
    for p in optimizer.param_groups:
        p["lr"] = parameters["learning_epoch"]

    if parameters["number_of_iterations"] < 0:
        iterations = len(data_loader)
    else:
        iterations = parameters["number_of_iterations"]

    model.train()

    for i, (image, target) in enumerate(data_loader):
        iterations_to_make_epoch = epoch * len(data_loader) + i
        if iterations_to_make_epoch <= parameters["iterations_to_warmup"]:
            r = iterations_to_make_epoch / parameters["iterations_to_warmup"]
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * parameters["learning_epoch"]

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        losses = model(image, target)
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iterations_to_make_epoch % parameters["publishing_losses_frequency"] == 0:
            print("{}\t".format(iterations_to_make_epoch),
                  "\t".join("{:.3f}".format(loss.item()) for loss in losses.values()))

        if i >= iterations - 1:
            break


def evaluate(model, data_loader, device, parameters):
    generate_results(model, data_loader, device, parameters)

    dataset = data_loader
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(parameters["result_path"], map_location="cpu")
    coco_evaluator.accumulate_results(results)

    temp = sys.stdout
    sys.stdout = CocoConversion()
    coco_evaluator.summarize()
    output = sys.stdout
    sys.stdout = temp

    return output


@torch.no_grad()
def generate_results(model, data_loader, device, parameters):
    if parameters["number_of_iterations"] < 0:
        iterations = len(data_loader)
    else:
        iterations = parameters["number_of_iterations"]
    ann_labels = data_loader.annotations_labels

    coco_results = []
    model.eval()
    for i, (image, target) in enumerate(data_loader):

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        #torch.cuda.synchronize()
        output = model(image)

        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction, ann_labels))

        if i >= iterations - 1:
            break

    torch.save(coco_results, parameters["result_path"])
