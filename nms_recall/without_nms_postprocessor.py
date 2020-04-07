import torch

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


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

    def forward_for_single_feature_map_without(self, anchors, box_cls, box_regression,
                                               pre_nms_thresh):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
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

        results = [[] for _ in range(N)]
        candidate_inds = box_cls > pre_nms_thresh

        for batch_idx, (per_box_cls, per_box_regression, per_candidate_inds, per_anchors) in enumerate(zip(
            box_cls,
            box_regression,
            candidate_inds,
            anchors
        )):
            # Sort and select TopN
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]
            per_class += 1

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

        return results

    def forward_without(self, anchors, box_cls, box_regression):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        anchors = list(zip(*anchors))
        for layer, (a, o, b) in enumerate(
            zip(anchors, box_cls, box_regression)):
            sampled_boxes.append(
                self.forward_for_single_feature_map_without(
                    a, o, b, self.pre_nms_thresh
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        return boxlists

    def forward_for_single_feature_map_with(self, anchors, box_cls, box_regression,
                                      pre_nms_thresh):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
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

        num_anchors = A * H * W

        results = [[] for _ in range(N)]
        candidate_inds = box_cls > pre_nms_thresh
        if candidate_inds.sum().item() == 0:
            empty_boxlists = []
            for a in anchors:
                empty_boxlist = BoxList(torch.Tensor(0, 4).to(device), a.size)
                empty_boxlist.add_field(
                    "labels", torch.LongTensor([]).to(device))
                empty_boxlist.add_field(
                    "scores", torch.Tensor([]).to(device))
                empty_boxlists.append(empty_boxlist)
            return empty_boxlists

        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

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

        return results

    def forward_with(self, anchors, box_cls, box_regression):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(box_cls)
        anchors = list(zip(*anchors))
        for layer, (a, o, b) in enumerate(
            zip(anchors, box_cls, box_regression)):
            sampled_boxes.append(
                self.forward_for_single_feature_map_with(
                    a, o, b,
                    self.pre_nms_thresh
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        boxlists = self.select_over_all_levels_with(boxlists)

        return boxlists

    def select_over_all_levels_with(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, 81):
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

    def forward(self, *args, **kwargs):
        return self.forward_with(*args, **kwargs), self.forward_without(*args, **kwargs)


def make_retinanet_postprocessor(
    config, fpn_post_nms_top_n, rpn_box_coder):

    pre_nms_thresh = 0.05
    # pre_nms_top_n = config.RETINANET.PRE_NMS_TOP_N
    # fpn_post_nms_top_n = fpn_post_nms_top_n

    pre_nms_top_n = 100000000000000
    fpn_post_nms_top_n = 1000000000000000

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
