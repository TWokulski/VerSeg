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
    base = base * 0.7 ** (n // 6)  # mask 0.8
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


def draw_image(image, target, classes=None, gt=False):
    image = image.clone()
    color = (0, 0, 255)
    image = image.clamp(0, 1)
    new_image = image.cpu().numpy()
    new_image = new_image.transpose(1, 2, 0)

    if target:
        box_image = np.zeros((646, 1096, 3))
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = xyxy2xywh(boxes).cpu().detach()
            for i, b in enumerate(boxes):
                b = b.tolist()
                b = list(map(int, b))
                p1 = (b[0], b[1])
                p2 = (b[0] + b[2], b[1] + b[3])
                box_image = cv2.rectangle(img=box_image, pt1=p1, pt2=p2, color=color, thickness=2)
            new_image += box_image

        masks = target["masks"]
        masks = masks.clamp(0, 1)
        mask_arr = masks.cpu().numpy()
        mask_image = np.zeros((646, 1096, 3))
        if gt:
            for m in mask_arr:
                m = np.array(m).reshape(646, 1096, 1)
                m = np.dstack((m, m, m))
                m[:, :] = m[:, :] * [255, 0, 0]
                mask_image += m
            new_image += mask_image
        else:
            index = 1
            for m in mask_arr:
                m = np.array(m).reshape(646, 1096, 1)
                if index < 81:
                    m = m * COLORS[index]/255
                else:
                    m = m * COLORS[index]/255
                index += 1
                mask_image += m
            new_image += mask_image
        cv2.imshow('image', new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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

'''
def draw_img(image, target, classes):
    image = image.clone()

    class_ids, classes, boxes, masks = [aa.cpu().numpy() for aa in results]
    num_detected = class_ids.shape[0]

    if num_detected == 0:
        return img_origin

    img_fused = img_origin
    if not cfg.hide_mask:
        masks_semantic = masks * (class_ids[:, None, None] + 1)  # expand class_ids' shape for broadcasting
        # The color of the overlap area is different because of the '%' operation.
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % (cfg.num_classes - 1)
        color_masks = COLORS[masks_semantic].astype('uint8')
        img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)

        if cfg.cutout:
            for i in range(num_detected):
                one_obj = np.tile(masks[i], (3, 1, 1)).transpose((1, 2, 0))
                one_obj = one_obj * img_origin
                new_mask = masks[i] == 0
                new_mask = np.tile(new_mask * 255, (3, 1, 1)).transpose((1, 2, 0))
                x1, y1, x2, y2 = boxes[i, :]
                img_matting = (one_obj + new_mask)[y1:y2, x1:x2, :]
                cv2.imwrite(f'results/images/{img_name}_{i}.jpg', img_matting)

    scale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    if not cfg.hide_bbox:
        for i in reversed(range(num_detected)):
            x1, y1, x2, y2 = boxes[i, :]

            color = COLORS[class_ids[i] + 1].tolist()
            cv2.rectangle(img_fused, (x1, y1), (x2, y2), color, thickness)

            class_name = cfg.class_names[class_ids[i]]
            text_str = f'{class_name}: {classes[i]:.2f}' if not cfg.hide_score else class_name

            text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
            cv2.rectangle(img_fused, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
            cv2.putText(img_fused, text_str, (x1, y1 + 15), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if cfg.real_time:
        fps_str = f'fps: {fps:.2f}'
        text_w, text_h = cv2.getTextSize(fps_str, font, scale, thickness)[0]
        # Create a shadow to show the fps more clearly
        img_fused = img_fused.astype(np.float32)
        img_fused[0:text_h + 8, 0:text_w + 8] *= 0.6
        img_fused = img_fused.astype(np.uint8)
        cv2.putText(img_fused, fps_str, (0, text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_fused
'''