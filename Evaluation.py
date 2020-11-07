import torch
import Mask_RCNN as algorithm

use_cuda = False
val_samples = 1
data_dir = 'Dataset'
num_classes = 2
model_path = 'ckpt/-2'

device = torch.device('cpu')

dataset = algorithm.COCODataset(data_dir, 'Validation', True)
classes = dataset.classes
coco = dataset.coco
iou_types = ['bbox', 'segm']
evaluator = algorithm.CocoEvaluator(coco, iou_types)

indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:val_samples])
model = algorithm.resnet50_for_mask_rcnn(True, num_classes).to(device)

user_model = torch.load(model_path, map_location=device)
model.load_state_dict(user_model['model'])
del user_model
model.eval()

for(image, target) in dataset:
    with torch.no_grad():
        result = model(image)
    result = {k: v.cpu() for k, v in result.items()}
    res = {target['image_id'].item(): result}
    evaluator.update(res)

    algorithm.show_single_target(image, target['masks'])
    # algorithm.show(image, result, classes)
