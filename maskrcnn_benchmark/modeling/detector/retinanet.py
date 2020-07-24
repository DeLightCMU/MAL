# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.retinanet import build_retinanet
from ..rpn.retinanetIA import build_retinanetIA
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import build_roi_mask_head
#from maskrcnn_benchmark.modeling.roi_heads.sparsemask_head.mask_head import build_sparse_mask_head
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import copy
import numpy as np
import math
from torch.autograd import Variable

import os
class RetinaNet(nn.Module):
    """
    Main class for RetinaNet
    It consists of three main parts:
    - backbone
    - bbox_heads: BBox prediction.
    - Mask_heads:
    """

    def __init__(self, cfg):
        super(RetinaNet, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.backbone = build_backbone(cfg)

        if self.cfg.FREEANCHOR.IA_ON:
            self.rpn = build_retinanetIA(cfg)
        else:
            self.rpn = build_retinanet(cfg)
        self.mask = None
        if cfg.MODEL.MASK_ON:
            self.mask = build_roi_mask_head(cfg)
        #if cfg.MODEL.SPARSE_MASK_ON:
        #    self.mask = build_sparse_mask_head(cfg)


    def forward(self, images, iteration=None, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        # Retina RPN Output
        rpn_features = features
        mask_all = []
        for rpn_feat in features:
            num_batch = rpn_feat.shape[0]
            num_channel = rpn_feat.shape[1]
            num_height = rpn_feat.shape[2]
            num_width = rpn_feat.shape[3]

            # compute cam with conv feat
            feat_channel_mean = torch.mean(rpn_feat.view(num_batch, num_channel, -1), dim=2)
            feat_channel_mean = feat_channel_mean.view(num_batch, num_channel, 1, 1)
            cam = torch.sum(rpn_feat * feat_channel_mean, 1)  # [B 1 H W]
            mask_all.append(cam)

        # Inverted Attention
        if self.cfg.FREEANCHOR.IA_ON and self.training and iteration is not None:
            rpn_features_tmp = []
            for feat_idx, rpn_feat in enumerate(rpn_features):
                rpn_features_tmp.append(rpn_feat.clone().detach())
            rpn_features_tmp = tuple(rpn_features_tmp)

            # the ratio of IA
            max_iteration = self.cfg.SOLVER.MAX_ITER
            ratio = self.ratio_function(self.cfg.FREEANCHOR.IA_TYPE, max_iteration, iteration)

            if self.cfg.FREEANCHOR.IA_FEAT:
                if self.cfg.FREEANCHOR.IA_FEAT_TYPE == 0:
                    mask = self.IA_feat(rpn_features_tmp, ratio)
                else:
                    mask = self.IA_feat2(rpn_features_tmp, ratio)
            else:
                mask = self.IA_grad(images, rpn_features_tmp, targets, ratio)

        if self.cfg.RETINANET.BACKBONE == "p2p7":
            rpn_features = features[1:]
        if self.cfg.FREEANCHOR.IA_ON and self.training:
            # print('images.size(): ', images.size(), targets)
            (anchors, detections), detector_losses = self.rpn(images, rpn_features, mask, targets=targets)
        else:
            (anchors, detections), detector_losses = self.rpn(images, rpn_features, targets=targets)
        # print('anchors: ', anchors)
        # print('detections: ', detections)
        # print('detector_losses: ', detector_losses)
        # print('size 1: ', images.size())
        # print('size 2: ', len(rpn_features))
        # for idx in range(len(rpn_features)):
        #     print('size 2: ', rpn_features[idx].size())
        # print('size 3: ', len(targets))
        # print('size 3: ', targets[0])

        if self.training:
            losses = {}
            losses.update(detector_losses)
            if self.mask:
                if self.cfg.MODEL.MASK_ON:
                    # Padding the GT
                    proposals = []
                    for (image_detections, image_targets) in zip(
                        detections, targets):
                        merge_list = []
                        if not isinstance(image_detections, list):
                            merge_list.append(image_detections.copy_with_fields('labels'))

                        if not isinstance(image_targets, list):
                            merge_list.append(image_targets.copy_with_fields('labels'))

                        if len(merge_list) == 1:
                            proposals.append(merge_list[0])
                        else:
                            proposals.append(cat_boxlist(merge_list))
                    x, result, mask_losses = self.mask(features, proposals, targets)
                    # print('x: ', x)
                    # print('result: ', result)
                    # print('mask_losses: ', mask_losses)
                elif self.cfg.MODEL.SPARSE_MASK_ON:
                    x, result, mask_losses = self.mask(features, anchors, targets)
                    # print('x: ', x)
                    # print('result: ', result)
                    # print('mask_losses: ', mask_losses)

                losses.update(mask_losses)
            return losses
        else:
            if self.mask:
                proposals = []
                for image_detections in detections:
                    num_of_detections = image_detections.bbox.shape[0]
                    if num_of_detections > self.cfg.RETINANET.NUM_MASKS_TEST > 0:
                        cls_scores = image_detections.get_field("scores")
                        image_thresh, _ = torch.kthvalue(
                            cls_scores.cpu(), num_of_detections - \
                            self.cfg.RETINANET.NUM_MASKS_TEST + 1
                        )
                        keep = cls_scores >= image_thresh.item()
                        keep = torch.nonzero(keep).squeeze(1)
                        image_detections = image_detections[keep]

                    proposals.append(image_detections)

                if self.cfg.MODEL.SPARSE_MASK_ON:
                    x, detections, mask_losses = self.mask(
                        features, proposals, targets
                    )
                else:
                    x, detections, mask_losses = self.mask(features, proposals, targets)
            return detections
        """
        return mask_all
        """

    def ratio_function(self, type, max_iteration, iteration):
        # 0: Constant 0.5; 1: Step (0.0-0.5); 2: Linear; 3: Log; 4: sigmoid
        # reflected 5: Step; 6: Linear; 7: Log; 8: sigmoid
        if type == 0:
            return 0.5
        if type == 1:
            if iteration < max_iteration/6 * 1:
                ratio = 0.0
            elif iteration < max_iteration/6 * 2:
                ratio = 0.1
            elif iteration < max_iteration/6 * 3:
                ratio = 0.2
            elif iteration < max_iteration/6 * 4:
                ratio = 0.3
            elif iteration < max_iteration / 6 * 5:
                ratio = 0.4
            else:
                ratio = 0.5
            return ratio
        if type == 2:
            return float(iteration) / float(max_iteration) / 2.0
        if type == 3:
            return math.log(1.0 + float(iteration) * 9.0 / float(max_iteration)) / 2.0
        if type == 4:
            return 1 / (1 + math.exp(-((float(iteration) - float(max_iteration)/2.0))*10.0/float(max_iteration))) / 2.0
        if type == 5:
            if iteration < max_iteration / 12 * 1:
                ratio = 0.0
            elif iteration < max_iteration / 12 * 2:
                ratio = 0.1
            elif iteration < max_iteration / 12 * 3:
                ratio = 0.2
            elif iteration < max_iteration / 12 * 4:
                ratio = 0.3
            elif iteration < max_iteration / 12 * 5:
                ratio = 0.4
            elif iteration < max_iteration / 12 * 6:
                ratio = 0.5
            elif iteration < max_iteration / 12 * 7:
                ratio = 0.5
            elif iteration < max_iteration / 12 * 8:
                ratio = 0.4
            elif iteration < max_iteration / 12 * 9:
                ratio = 0.3
            elif iteration < max_iteration / 12 * 10:
                ratio = 0.2
            elif iteration < max_iteration / 12 * 11:
                ratio = 0.1
            else:
                ratio = 0.0
            return ratio
        if type == 6:
            if iteration < max_iteration/2:
                return float(iteration) / float(max_iteration)
            else:
                return float(max_iteration - iteration) / float(max_iteration)
        if type == 7:
            if iteration < max_iteration/2:
                return math.log(1.0 + float(iteration) * 18.0 / float(max_iteration)) / 2.0
            else:
                return math.log(1.0 + float(max_iteration - iteration) * 18.0 / float(max_iteration)) / 2.0
        if type == 8:
            if iteration < max_iteration/2:
                return 1 / (1 + math.exp(-((float(iteration) - float(max_iteration)/2.0))*10.0/float(max_iteration))) / 2.0
            else:
                return 1 / (1 + math.exp(-((float(max_iteration - iteration) - float(max_iteration)/2.0))*10.0/float(max_iteration))) / 2.0
        if type == 9:
            if iteration < 30000:
                ratio = 0.0
            elif iteration < 60000:
                ratio = 0.5
            elif iteration < 90000:
                ratio = 0.0
            elif iteration < 105000:
                ratio = 0.0
            elif iteration < 120000:
                ratio = 0.5
            else:
                ratio = 0.0
            return ratio

    def IA_feat(self, feature, ratio):
        """
            feature: rpn features
            ratio: how many to be inverted
        """
        mask_all = []
        for rpn_feat in feature:
            num_batch = rpn_feat.shape[0]
            num_channel = rpn_feat.shape[1]
            num_height = rpn_feat.shape[2]
            num_width = rpn_feat.shape[3]

            # compute cam with conv feat
            feat_channel_mean = torch.mean(rpn_feat.view(num_batch, num_channel, -1), dim=2)
            feat_channel_mean = feat_channel_mean.view(num_batch, num_channel, 1, 1)
            cam = torch.sum(rpn_feat * feat_channel_mean, 1) #[B 1 H W]

            # threshold
            cam_tmp = cam.view(num_batch, num_height*num_width)
            th_idx = int(num_height * num_width * ratio)
            th25_mask_value = torch.sort(cam_tmp, dim=1, descending=True)[0][:, th_idx]
            th25_mask_value = th25_mask_value.view(num_batch, 1).expand(num_batch, num_height*num_width)
            th25_mask_value = th25_mask_value.reshape(num_batch, num_height, num_width)
            mask_all_cuda = torch.where(cam > th25_mask_value, torch.zeros(cam.shape).cuda(),
                                    torch.ones(cam.shape).cuda())
            mask_all_cuda = mask_all_cuda.reshape(num_batch, 1, num_height, num_width)
            mask_all.append(mask_all_cuda.detach())  # [256, 49]

        return mask_all

    def IA_feat2(self, feature, ratio):
        """
            feature: rpn features
            ratio: how many to be inverted
        """
        mask_all = []
        for rpn_feat in feature:
            num_batch = rpn_feat.shape[0]
            num_channel = rpn_feat.shape[1]
            num_height = rpn_feat.shape[2]
            num_width = rpn_feat.shape[3]

            # compute cam with conv feat
            gap = torch.mean(rpn_feat.view(num_batch, num_channel, -1), dim=2)
            gap = gap.view(num_batch, num_channel, 1, 1)

            # channel-wise IA
            gap_tmp = gap.view(num_batch, num_channel)
            th_idx = int(num_channel * ratio)
            th_channel_mask_value = torch.sort(gap_tmp, dim=1, descending=True)[0][:, th_idx]
            th_channel_mask_value = th_channel_mask_value.view(num_batch, 1).expand(num_batch, num_channel)
            mask_channel_cuda = torch.where(gap_tmp > th_channel_mask_value, torch.zeros(gap_tmp.shape).cuda(),
                                        torch.ones(gap_tmp.shape).cuda())
            mask_channel_cuda = mask_channel_cuda.view(num_batch, num_channel, 1, 1).expand(num_batch, num_channel, num_height, num_width)

            cam = torch.sum(rpn_feat * gap, 1) #[B 1 H W]
            # threshold -- spatial wise IA
            cam_tmp = cam.view(num_batch, num_height*num_width)
            th_idx = int(num_height * num_width * ratio)
            th25_mask_value = torch.sort(cam_tmp, dim=1, descending=True)[0][:, th_idx]
            th25_mask_value = th25_mask_value.view(num_batch, 1).expand(num_batch, num_height*num_width)
            th25_mask_value = th25_mask_value.reshape(num_batch, num_height, num_width)
            mask_sptial_cuda = torch.where(cam > th25_mask_value, torch.zeros(cam.shape).cuda(),
                                    torch.ones(cam.shape).cuda())
            mask_sptial_cuda = mask_sptial_cuda.reshape(num_batch, 1, num_height, num_width)
            mask_sptial_cuda = mask_sptial_cuda.expand(num_batch, num_channel, num_height, num_width)

            mask_cuda = mask_sptial_cuda * mask_channel_cuda
            mask_all.append(mask_cuda.detach())  # [256, 49]

        return mask_all

    def IA_grad(self, images, features, targets, ratio):
        self.eval()
        rpn_features = []
        for idx, rpn_feat in enumerate(features):
            rpn_feat.requires_grad = True
            rpn_features.append(rpn_feat)
        features = tuple(rpn_features)
        (anchors, detections), detector_losses = self.rpn(images, features, None, gflag=True, targets=targets)

        losses = sum(loss for loss in detector_losses.values())
        self.zero_grad()
        losses.backward()
        grads_val = []
        for idx, rpn_feat in enumerate(features):
            grads_val.append(rpn_feat.grad.clone().detach())

        if self.cfg.FREEANCHOR.IA_FEAT_TYPE == 0:
            mask_all = self.IA_feat(grads_val, ratio)
        else:
            mask_all = self.IA_feat2(grads_val, ratio)

        self.train()
        return mask_all




