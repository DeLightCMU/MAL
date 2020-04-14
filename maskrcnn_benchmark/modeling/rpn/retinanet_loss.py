"""
This file contains specific functions for computing losses on the RetinaNet
file
"""

import torch
from torch.nn import functional as F

from ..utils import cat

from maskrcnn_benchmark.layers import SmoothL1Loss
from maskrcnn_benchmark.layers import AdjustSmoothL1Loss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


class RetinaNetLossComputation(object):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, cfg, proposal_matcher, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.num_classes = cfg.RETINANET.NUM_CLASSES -1
        self.box_cls_loss_func = SigmoidFocalLoss(
            self.num_classes,
            cfg.RETINANET.LOSS_GAMMA,
            cfg.RETINANET.LOSS_ALPHA
        )
        if cfg.RETINANET.SELFADJUST_SMOOTH_L1:
            self.regression_loss = AdjustSmoothL1Loss(
                4,
                beta=cfg.RETINANET.BBOX_REG_BETA
            )
        else:
            self.regression_loss = SmoothL1Loss(
                beta=cfg.RETINANET.BBOX_REG_BETA
            )

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(['labels'])
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_targets.get_field("labels").clone()

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard indices that are between thresholds 
            # -1 will be ignored in SigmoidFocalLoss
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            labels_per_image = labels_per_image.to(dtype=torch.float32)
            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, box_cls, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        # sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        # sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        # sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        num_layers = len(box_cls)
        box_cls_flattened = []
        box_regression_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for box_cls_per_level, box_regression_per_level in zip(
            box_cls, box_regression
        ):
            N, A, H, W = box_cls_per_level.shape
            C = self.num_classes
            box_cls_per_level = box_cls_per_level.view(N, -1, C, H, W)
            box_cls_per_level = box_cls_per_level.permute(0, 3, 4, 1, 2)
            box_cls_per_level = box_cls_per_level.reshape(N, -1, C)
            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)
            box_cls_flattened.append(box_cls_per_level)
            box_regression_flattened.append(box_regression_per_level)
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
        box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = labels > 0

        retinanet_regression_loss = self.regression_loss(
            box_regression[pos_inds],
            regression_targets[pos_inds],
            size_average=False,
        ) / (pos_inds.sum() * 4)
        labels = labels.int()

        retinanet_cls_loss =self.box_cls_loss_func(
            box_cls,
            labels
        ) / ((labels > 0).sum() + N)

        losses = {
            "loss_retina_cls": retinanet_cls_loss,
            "loss_retina_reg": retinanet_regression_loss,
        }

        return losses


def make_retinanet_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=cfg.RETINANET.LOW_QUALITY_MATCHES,
        low_quality_threshold=cfg.RETINANET.LOW_QUALITY_THRESHOLD
    )

    loss_evaluator = RetinaNetLossComputation(
        cfg, matcher, box_coder
    )
    return loss_evaluator
