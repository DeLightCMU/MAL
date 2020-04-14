import torch
import time
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
from maskrcnn_benchmark.layers import nv_decode as _nv_decode
from maskrcnn_benchmark.layers import nv_nms as _nv_nms

class RetinaNetPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        box_coder=None,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            box_coder (BoxCoder)
        """
        super(RetinaNetPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cell_anchors = [[[-18.0000,  -8.0000,  25.0000,  15.0000], [-23.7183, -11.1191,  30.7183,  18.1191], [-30.9228, -15.0488,  37.9228,  22.0488], \
                 [-12.0000, -12.0000,  19.0000,  19.0000], [-16.1587, -16.1587,  23.1587,  23.1587], [-21.3984, -21.3984,  28.3984,  28.3984], \
                 [ -8.0000, -20.0000,  15.0000,  27.0000], [-11.1191, -26.2381,  18.1191,  33.2381], [-15.0488, -34.0976,  22.0488,  41.0976]], \
                [[-38.0000, -16.0000, 53.0000, 31.0000], [-49.9564, -22.2381, 64.9564, 37.2381], [-65.0204, -30.0976, 80.0204, 45.0976], \
                 [-24.0000, -24.0000, 39.0000, 39.0000], [-32.3175, -32.3175, 47.3175, 47.3175], [-42.7968, -42.7968, 57.7968, 57.7968], \
                 [-14.0000, -36.0000, 29.0000, 51.0000], [-19.7183, -47.4365, 34.7183, 62.4365], [-26.9228, -61.8456, 41.9228, 76.8456]], \
                [[-74.0000, -28.0000, 105.0000, 59.0000], [-97.3929, -39.4365, 128.3929, 70.4365], [-126.8661, -53.8456, 157.8661, 84.8456], \
                 [-48.0000, -48.0000, 79.0000, 79.0000], [-64.6349, -64.6349, 95.6349, 95.6349], [-85.5937, -85.5937, 116.5937, 116.5937], \
                 [-30.0000, -76.0000, 61.0000, 107.0000], [-41.9564, -99.9127, 72.9564, 130.9127], [-57.0204, -130.0409, 88.0204, 161.0409]], \
                [[-150.0000, -60.0000, 213.0000, 123.0000], [-197.3056, -83.9127, 260.3056, 146.9127], [-256.9070, -114.0409, 319.9070, 177.0409], \
                 [-96.0000, -96.0000, 159.0000, 159.0000], [-129.2699, -129.2699, 192.2699, 192.2699], [-171.1873, -171.1873, 234.1873, 234.1873], \
                 [-58.0000, -148.0000, 121.0000, 211.0000], [-81.3929, -194.7858, 144.3929, 257.7858], [-110.8661, -253.7322, 173.8661, 316.7322]], \
                [[-298.0000, -116.0000, 425.0000, 243.0000], [-392.0914, -162.7858, 519.0914, 289.7858], [-510.6392, -221.7322, 637.6392, 348.7322], \
                 [-192.0000, -192.0000, 319.0000, 319.0000], [-258.5398, -258.5398, 385.5398, 385.5398], [-342.3747, -342.3747, 469.3747, 469.3747], \
                 [-118.0000, -300.0000, 245.0000, 427.0000], [-165.3056, -394.6113, 292.3056, 521.6113], [-224.9070, -513.8140, 351.9070, 640.8140]] \
                ]
        self.strides = [8, 16, 32, 64, 128]

    def forward_for_single_feature_map(self, anchors, box_cls, box_regression,
                                      pre_nms_thresh, stride):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        pre_time = time.time()

        device = box_cls.device
        N, _ , H, W = box_cls.shape
        A = int(box_regression.size(1) / 4)
        C = int(box_cls.size(1) / A)

        # put in the same format as anchors
        box_cls = box_cls.view(N, -1, C, H, W).permute(0, 3, 4, 1, 2)
        box_cls = box_cls.reshape(N, -1, C)
        box_cls = box_cls.sigmoid()

        box_regression = box_regression.view(N, -1, 4, H, W)
        box_regression = box_regression.permute(0, 3, 4, 1, 2)
        box_regression = box_regression.reshape(N, -1, 4)

        # print('nms-singlelayer-preprocessing stage1 takes {}'.format(time.time() - pre_time))
        pre_time = time.time()

        num_anchors = A * H * W

        results = [[] for _ in range(N)]


        candidate_inds = box_cls > pre_nms_thresh

        # print('nms-singlelayer-preprocessing stage2-1 takes {}'.format(time.time() - pre_time))
        pre_time = time.time()

        if candidate_inds.sum().item() == 0:
            empty_boxlists = []
            print('if cmd')
            for a in anchors:
                empty_boxlist = BoxList(torch.Tensor(0, 4).to(device), a.size)
                empty_boxlist.add_field(
                    "labels", torch.LongTensor([]).to(device))
                empty_boxlist.add_field(
                    "scores", torch.Tensor([]).to(device))
                empty_boxlists.append(empty_boxlist)
            return empty_boxlists

        # print('nms-singlelayer-preprocessing stage2-2 takes {}'.format(time.time() - pre_time))
        pre_time = time.time()

        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # print('nms-singlelayer-preprocessing stage3 takes {}'.format(time.time() - pre_time))
        pre_time = time.time()

        for batch_idx, (per_box_cls, per_box_regression, per_pre_nms_top_n, \
        per_candidate_inds, per_anchors) in enumerate(zip(
            box_cls,
            box_regression,
            pre_nms_top_n,
            candidate_inds,
            anchors)):

            # Sort and select TopN
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]
            per_class += 1
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                        per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_box_loc = per_box_loc[top_k_indices]
                per_class = per_class[top_k_indices]

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results[batch_idx] = boxlist

        # print('nms-singlelayer-postrocessing takes {}'.format(time.time() - pre_time))
        pre_time = time.time()

        return results

    def forward_for_single_feature_map1(self, pre_anchors, box_cls, box_regression,
                                      pre_nms_thresh, stride):
        """
        retinanet-example
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        anchors = torch.Tensor(self.cell_anchors[self.strides.index(stride)])
        top_n = self.pre_nms_top_n
        batch_size = box_cls.size()[0]
        device = box_cls.device
        box_cls = box_cls.sigmoid()

        out_scores = torch.zeros((batch_size, top_n), device=device)
        out_boxes = torch.zeros((batch_size, top_n, 4), device=device)
        out_classes = torch.zeros((batch_size, top_n), device=device).long()
        results = [[] for _ in range(batch_size)]
        if torch.cuda.is_available() and 0:
            out_scores, out_boxes, out_classes =  _nv_decode(box_cls.float(), box_regression.float(),
                stride, pre_nms_thresh, top_n, anchors.view(-1).tolist())
            out_classes = out_classes.long()
            out_classes = out_classes + 1
        else:
            anchors = anchors.to(device).type(box_cls.type())
            num_anchors = anchors.size()[0] if anchors is not None else 1
            num_classes = box_cls.size()[1] // num_anchors
            height, width = box_cls.size()[-2:]


            # Per item in batch
            for batch in range(batch_size):
                cls_head = box_cls[batch, :, :, :].contiguous().view(-1)
                box_head = box_regression[batch, :, :, :].contiguous().view(-1, 4)

                # Keep scores over threshold
                keep = (cls_head >= pre_nms_thresh).nonzero().view(-1)
                if keep.nelement() == 0:
                    empty_boxlists = []
                    for a in pre_anchors:
                        empty_boxlist = BoxList(torch.Tensor(0, 4).to(device), a.size)
                        empty_boxlist.add_field(
                            "labels", torch.LongTensor([]).to(device))
                        empty_boxlist.add_field(
                            "scores", torch.Tensor([]).to(device))
                        empty_boxlists.append(empty_boxlist)
                    return empty_boxlists

                # Gather top elements
                scores = torch.index_select(cls_head, 0, keep)
                scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
                indices = torch.index_select(keep, 0, indices).view(-1)
                classes = (indices / width / height) % num_classes
                classes = classes.long()
                classes = classes + 1

                # Infer kept bboxes
                x = indices % width
                y = (indices / width) % height
                a = indices / num_classes / height / width
                box_head = box_head.view(num_anchors, 4, height, width)
                boxes = box_head[a, :, y, x]

                if anchors is not None:
                    grid = torch.stack([x, y, x, y], 1).type(box_cls.type()) * stride + anchors[a, :]
                    boxes = self.box_coder.decode(boxes, grid)

                out_scores[batch, :scores.size()[0]] = scores
                out_boxes[batch, :boxes.size()[0], :] = boxes
                out_classes[batch, :classes.size()[0]] = classes

        for batch in range(batch_size):
            boxlist = BoxList(out_boxes[batch], pre_anchors[batch].size, mode="xyxy")
            boxlist.add_field("labels", out_classes[batch])
            boxlist.add_field("scores", out_scores[batch])
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results[batch] = boxlist

        return results

    def forward(self, anchors, box_cls, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        """
        if torch.cuda.is_available():
            box_cls = [cls_head.sigmoid() for cls_head in box_cls]
            decoded = []
            for stride, cls_head, box_head in zip(self.strides, box_cls, box_regression):
                self.anchors[stride] = torch.Tensor(self.cell_anchors[self.strides.index(stride)])

                # Decode and filter boxes
                decoded.append(_nv_decode(cls_head, box_head, stride,
                                      self.threshold, self.top_n, self.anchors[stride]))

            # Perform non-maximum suppression
            decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]

            # scores, boxes, classes
            results = _nv_nms(*decoded, self.nms_thresh, self.detections)
        """
        sampled_boxes = []
        num_levels = len(box_cls)
        anchors = list(zip(*anchors))

        strides = self.strides
        for layer, (a, o, b, s) in enumerate(
            zip(anchors, box_cls, box_regression, strides)):
            sampled_boxes.append(
                self.forward_for_single_feature_map1(
                    a, o, b,
                    self.pre_nms_thresh, s
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        boxlists = self.select_over_all_levels(boxlists)
        # print('nms-postprocessing takes {}'.format(time.time() - pre_time))
        pre_time = time.time()

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            # for j in range(1, 81):
            for j in [1,3,6,8]:
                inds = (labels == j).nonzero().view(-1)
                if len(inds) == 0:
                    continue

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms_thresh,
                    score_field="scores"
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            if len(result) > 0:
                result = cat_boxlist(result)
                number_of_detections = len(result)

                # Limit to max_per_image detections **over all classes**
                if number_of_detections > self.fpn_post_nms_top_n > 0:
                    cls_scores = result.get_field("scores")
                    image_thresh, _ = torch.kthvalue(
                        cls_scores.cpu(),
                        number_of_detections - self.fpn_post_nms_top_n + 1
                    )
                    keep = cls_scores >= image_thresh.item()
                    keep = torch.nonzero(keep).squeeze(1)
                    result = result[keep]
                results.append(result)
            else:
                empty_boxlist = BoxList(torch.zeros(1, 4).to('cuda'), boxlist.size)
                empty_boxlist.add_field(
                    "labels", torch.LongTensor([1]).to('cuda'))
                empty_boxlist.add_field(
                    "scores", torch.Tensor([0.01]).to('cuda'))
                results.append(empty_boxlist)
        return results


def make_retinanet_postprocessor(
    config, fpn_post_nms_top_n, rpn_box_coder):

    pre_nms_thresh = 0.05
    pre_nms_top_n = config.RETINANET.PRE_NMS_TOP_N
    fpn_post_nms_top_n = fpn_post_nms_top_n
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RetinaNetPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        box_coder=rpn_box_coder,
        min_size=min_size
    )
    return box_selector
