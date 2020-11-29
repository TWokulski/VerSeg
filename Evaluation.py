import torch
import Mask_RCNN as algorithm
import cv2

use_cuda = False
val_samples = 9
data_dir = 'Dataset'
num_classes = 2
model_path = 'ckpt/bestbest-98.pth'

device = torch.device('cpu')

dataset = algorithm.COCODataset(data_dir, 'Validation', True)
classes = dataset.classes
coco = dataset.coco
iou_types = ['bbox', 'segm']
evaluator = algorithm.CocoEvaluator(coco, iou_types)

your_dataset_indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, your_dataset_indices[:val_samples])
model = algorithm.resnet50_for_mask_rcnn(True, num_classes).to(device)

user_model = torch.load(model_path, map_location=device)
model.load_state_dict(user_model['model'])
del user_model
model.eval()
preds = []
i = 0
for(image, target) in dataset:
    with torch.no_grad():
        result = model(image)
    result = {k: v.cpu() for k, v in result.items()}
    res = {target['image_id'].item(): result}
    evaluator.update(res)
    # print(target['boxes'])
    b = algorithm.xyxy2xywh(target['boxes'])
    # print(b.cpu().detach())
    # for i, j in enumerate(b):
        # print(i, j)
    # algorithm.draw_image(image, target, classes, False)
    original_img, gt_box, gt_mask, pred_box, pred_mask = algorithm.draw_image(image, target, result, classes)
    pred = original_img + pred_box + pred_mask
    preds.append(pred)
    # cv2.imshow('original_img', original_img)
    # cv2.imshow('gt_box', gt_box)
    # cv2.imshow('gt_mask', gt_mask)
    # cv2.imshow('pred_box', pred_box)
    # cv2.imshow('pred_mask', pred_mask)
# for p in preds:

cv2.imshow('p', preds[0])

cv2.waitKey(0)
cv2.destroyAllWindows()

    # algorithm.show_single_target(image, target['masks'], target['boxes'])
    #algorithm.show_single(image, result, classes)
