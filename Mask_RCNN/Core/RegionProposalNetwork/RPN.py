import torch
from torch import nn
import torch.nn.functional as F
from ..Utils import box_iou
from ..Utils import BoxCoder
from ..Utils import Matcher
from torchvision.ops.boxes import nms
from ..Utils import BalancedPositiveNegativeSampler


class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head, fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction, reg_weights, proposals_before_nms,
                 proposals_after_nms, nms_thresh):

        super().__init__()

        self.anchor_gen = anchor_generator
        self.head = head
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh)
        self.sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        self._proposals_before_nms = proposals_before_nms
        self._proposals_after_nms = proposals_after_nms
        self.nms_thresh = nms_thresh
        self.min_size = 1

    def create_proposal(self, anchor, classifier, delta, image_shape):
        if self.training:
            proposals_before_nms = self._proposals_before_nms['training']
            proposals_after_nms = self._proposals_after_nms['training']
        else:
            proposals_before_nms = self._proposals_before_nms['testing']
            proposals_after_nms = self._proposals_after_nms['testing']

        proposals_before_nms = min(classifier.shape[0], proposals_before_nms)
        top_n_idx = classifier.topk(proposals_before_nms)[1]
        score = classifier[top_n_idx]
        proposal = self.box_coder.decode(delta[top_n_idx], anchor[top_n_idx])

        proposal, score = self.box_coder.process_box(proposal, score, image_shape, self.min_size)
        keep = nms(proposal, score, self.nms_thresh)[:proposals_after_nms]
        proposal = proposal[keep]
        return proposal

    def compute_loss(self, classifier, delta, gt_box, anchor):
        iou = box_iou(gt_box, anchor)
        label, matched_idx = self.proposal_matcher(iou)

        pos_idx, neg_idx = self.fg_bg_sampler(label)
        idx = torch.cat((pos_idx, neg_idx))
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], anchor[pos_idx])

        classifier_loss = F.binary_cross_entropy_with_logits(classifier[idx], label[idx])
        box_loss = F.l1_loss(delta[pos_idx], regression_target, reduction='sum') / idx.numel()

        return classifier_loss, box_loss

    def forward(self, feature, image_shape, target=None):
        if target is not None:
            gt_box = target['boxes']
        anchor = self.anchor_generator(feature, image_shape)

        classifier, delta = self.head(feature)
        classifier = classifier.permute(0, 2, 3, 1).flatten()
        delta = delta.permute(0, 2, 3, 1).reshape(-1, 4)

        proposal = self.create_proposal(anchor, classifier.detach(), delta.detach(), image_shape)
        if self.training:
            classifier_loss, box_loss = self.compute_loss(classifier, delta, gt_box, anchor)

            return proposal, dict(rpn_objectness_loss=classifier_loss, rpn_box_loss=box_loss)

        return proposal, {}
