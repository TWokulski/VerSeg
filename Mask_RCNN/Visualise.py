import torch
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from Config import COLORS
import cv2


def xyxy2xywh(box):
    new_box = torch.zeros_like(box)
    new_box[:, 0] = box[:, 0]
    new_box[:, 1] = box[:, 1]
    new_box[:, 2] = box[:, 2] - box[:, 0]
    new_box[:, 3] = box[:, 3] - box[:, 1]
    return new_box


def factor(n, base=1):
    base = base * 0.7 ** (n // 6)
    i = n % 6
    if i < 3:
        f = [0, 0, 0]
        f[i] = base
    else:
        base /= 2
        f = [base, base, base]
        f[i - 3] = 0
    return f


def show_prediction(images, targets=None, classes=None):
    if isinstance(images, (list, tuple)):
        for i in range(len(images)):
            show_single(images[i], targets[i] if targets else targets, classes)
    else:
        show_single(images, targets, classes)


def show_target(images, targets):
    if isinstance(images, (list, tuple)):
        for i in range(len(images)):
            show_single_target(images[i], targets[i])
    else:
        show_single_target(images, targets)


def show_single(image, target, classes):
    image = image.clone()
    if target and "masks" in target:
        masks = target["masks"].unsqueeze(1)
        masks = masks.repeat(1, 3, 1, 1)
        for i, m in enumerate(masks):
            f = torch.tensor(factor(i, 0.4)).reshape(3, 1, 1).to(image)
            value = f * m
            image += value

    ax = plt.subplot(111)
    image = image.clamp(0, 1)
    im = image.cpu().numpy()
    ax.imshow(im.transpose(1, 2, 0))  # RGB
    H, W = image.shape[-2:]
    ax.set_title("H: {}   W: {}".format(H, W))
    ax.axis("off")

    if target:
        if "labels" in target:
            if classes is None:
                raise ValueError("'classes' should not be None when 'target' has 'labels'!")
            tags = {l: i for i, l in enumerate(tuple(set(target["labels"].tolist())))}

        index = 0
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = xyxy2xywh(boxes).cpu().detach()
            for i, b in enumerate(boxes):
                if "labels" in target:
                    l = target["labels"][i].item()
                    index = tags[l]
                    txt = classes[l]
                    if "scores" in target:
                        s = target["scores"][i]
                        s = round(s.item() * 100)
                        txt = "{} {}%".format(txt, s)
                    ax.text(
                        b[0], b[1], txt, fontsize=9, color=(1, 1, 1),
                        horizontalalignment="left", verticalalignment="bottom",
                        bbox=dict(boxstyle="square", fc="black", lw=1, alpha=1)
                    )
                rect = patches.Rectangle(b[:2], b[2], b[3], linewidth=2, edgecolor=factor(index), facecolor="none")
                ax.add_patch(rect)
    plt.show()


def draw_image(image, target=None, result=None, classes=None):
    image = image.clone()
    image = image.clamp(0, 1)
    new_image = image.cpu().numpy()
    new_image = new_image.transpose(1, 2, 0)
    shape = new_image.shape

    if target:
        if "boxes" in target:
            boxes = target["boxes"]
            boxes_image_gt = get_boxes_array(boxes, shape)
        else:
            boxes_image_gt = np.zeros(shape)

        if "masks" in target:
            masks = target["masks"]
            mask_image_gt = get_mask_array(masks, True, shape)
        else:
            mask_image_gt = np.zeros(shape)

    if result:
        if "boxes" in result:
            boxes = result["boxes"]
            boxes_image_result = get_boxes_array(boxes, shape)
        else:
            boxes_image_result = np.zeros(shape)

        if "masks" in result:
            masks = result["masks"]
            mask_image_result = get_mask_array(masks, False, shape)
        else:
            mask_image_result = np.zeros(shape)

    return (new_image*255).astype("uint8"), boxes_image_gt, mask_image_gt, boxes_image_result, mask_image_result


def get_boxes_array(boxes, shape):
    color = (255, 0, 0)
    box_image = np.zeros(shape)
    boxes = xyxy2xywh(boxes).cpu().detach()
    for i, b in enumerate(boxes):
        b = b.tolist()
        b = list(map(int, b))
        p1 = (b[0], b[1])
        p2 = (b[0] + b[2], b[1] + b[3])
        box_image = cv2.rectangle(img=box_image, pt1=p1, pt2=p2, color=color, thickness=2)

    return box_image.astype("uint8")


def get_mask_array(masks, gt, shape):
    masks = masks.clamp(0, 1)
    mask_arr = masks.cpu().numpy()
    mask_image = np.zeros(shape)

    if gt:
        for m in mask_arr:
            m = np.array(m).reshape(shape[0], shape[1], 1)
            m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
            m = m * [255, 0, 0]
            mask_image += m
        return mask_image.astype("uint8")
    else:
        index = 1
        for m in mask_arr:
            m = np.array(m).reshape(shape[0], shape[1], 1)
            if index < 81:
                m = m * COLORS[index]
            else:
                m = m * COLORS[index - 80]
            index += 1
            mask_image += m
        return mask_image.astype("uint8")


def show_single_target(image, target, bouding):
    target = target.clone()
    image = image.clone()

    image = image.clamp(0, 1)
    new_image = image.cpu().numpy()
    new_image = new_image.transpose(1, 2, 0)

    #ax = plt.subplot()
    target = target.clamp(0, 1)
    mask_arr = target.cpu().numpy()
    mask = np.zeros((646, 1096, 3))
    for t in mask_arr:
        t = np.array(t).reshape(646, 1096, 1)
        # t = np.dstack((t, t, t))
        t = t * COLORS[57]/255
        print(t.shape)
        mask += t
    new_image += mask
    color = (255, 0, 0)
    boxes = xyxy2xywh(bouding).cpu().detach()
    box = np.zeros((646, 1096, 3))
    for i, b in enumerate(boxes):
        b = b.tolist()
        b = list(map(int, b))
        p1 = (b[0], b[1])
        p2 = (b[0] + b[2], b[1] + b[3])
        box = cv2.rectangle(img=box, pt1=p1, pt2=p2, color=color, thickness=2)
    new_image += box

    cv2.imshow('image', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #plt.imshow(new_image)
    #plt.axis("off")

    #plt.savefig('fig.png', bbox_inches='tight')

