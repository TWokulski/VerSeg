from torch import nn
from torch.utils.model_zoo import load_url
from Mask_RCNN.Core.RegionProposalNetwork import AnchorGenerator
from Mask_RCNN.Core.RegionProposalNetwork import RPNHead
from Mask_RCNN.Core.RegionProposalNetwork import RegionProposalNetwork
from Mask_RCNN.Core.RegionOfInterest import RoIAlign
from Mask_RCNN.Core.RegionOfInterest import RoIHeads
from Mask_RCNN.Core.Predictors import ResBackbone
from Mask_RCNN.Core.Predictors import FastRCNNPredictor
from Mask_RCNN.Core.Predictors import MaskRCNNPredictor
from Mask_RCNN.Core.Utils import Processing
from Mask_RCNN.Core.MaskRCNN_Config import *


class MaskRCNN(nn.Module):

    def __init__(self, backbone, number_of_classes):

        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels

        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_num_samples, rpn_positive_fraction,
            rpn_reg_weights,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, number_of_classes)

        self.head = RoIHeads(
            box_roi_pool, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_num_samples, box_positive_fraction,
            box_reg_weights,
            box_score_thresh, box_nms_thresh, box_num_detections)

        self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)

        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, number_of_classes)

        self.transformer = Processing(
            min_size=800, max_size=1333,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225])

    def forward(self, image, target=None):
        original_image_shape = image.shape[-2:]

        image = self.transformer.normalize(image)
        image, target = self.transformer.resize(image, target)
        image = self.transformer.batched_image(image)

        image_shape = image.shape[-2:]
        feature = self.backbone(image)

        proposal, rpn_losses = self.rpn(feature, image_shape, target)
        result, roi_losses = self.head(feature, proposal, image_shape, target)

        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            result = self.transformer.post_processing(result, image_shape, original_image_shape)
            return result


def resnet50_for_mask_rcnn(use_pre_trained, number_of_classes):
    backbone = ResBackbone('resnet50')
    model = MaskRCNN(backbone, number_of_classes)

    if use_pre_trained:
        model_state_dict = load_url(model_url['maskrcnn_resnet50_fpn_coco'])
        pre_trained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]

        for i, del_idx in enumerate(del_list):
            pre_trained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        for i, name in enumerate(msd):
            if i in skip_list:
                continue
            msd[name].copy_(pre_trained_msd[i])

        model.load_state_dict(msd)

    return model
