// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "SigmoidFocalLoss.h"

vector<at::Tensor> nv_decode(at::Tensor cls_head, at::Tensor box_head,
        int scale, float score_thresh, int top_n, vector<float> &anchors) {

    CHECK_INPUT(cls_head);
    CHECK_INPUT(box_head);

    int batch = cls_head.size(0);
    int num_anchors = anchors.size() / 4;
    int num_classes = cls_head.size(1) / num_anchors;
    int height = cls_head.size(2);
    int width = cls_head.size(3);
    auto options = cls_head.options();

    auto scores = at::zeros({batch, top_n}, options);
    auto boxes = at::zeros({batch, top_n, 4}, options);
    auto classes = at::zeros({batch, top_n}, options);


    // Create scratch buffer
    int size = retinanet::cuda::decode(batch, nullptr, nullptr, height, width, scale,
        num_anchors, num_classes, anchors, score_thresh, top_n, nullptr, 0, nullptr);
    auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

    // Decode boxes
    vector<void *> inputs = {cls_head.data_ptr(), box_head.data_ptr()};
    vector<void *> outputs = {scores.data_ptr(), boxes.data_ptr(), classes.data_ptr()};
    retinanet::cuda::decode(batch, inputs.data(), outputs.data(), height, width, scale,
        num_anchors, num_classes, anchors, score_thresh, top_n,
        scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());

    return {scores, boxes, classes};
}

vector<at::Tensor> nv_nms(at::Tensor scores, at::Tensor boxes, at::Tensor classes,
        float nms_thresh, int detections_per_im) {

    CHECK_INPUT(scores);
    CHECK_INPUT(boxes);
    CHECK_INPUT(classes);

    int batch = scores.size(0);
    int count = scores.size(1);
    auto options = scores.options();

    auto nms_scores = at::zeros({batch, detections_per_im}, scores.options());
    auto nms_boxes = at::zeros({batch, detections_per_im, 4}, boxes.options());
    auto nms_classes = at::zeros({batch, detections_per_im}, classes.options());

    // Create scratch buffer
    int size = retinanet::cuda::nms(batch, nullptr, nullptr, count, 
        detections_per_im, nms_thresh, nullptr, 0, nullptr);
    auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

    // Perform NMS
    vector<void *> inputs = {scores.data_ptr(), boxes.data_ptr(), classes.data_ptr()};
    vector<void *> outputs = {nms_scores.data_ptr(), nms_boxes.data_ptr(), nms_classes.data_ptr()};
    retinanet::cuda::nms(batch, inputs.data(), outputs.data(), count,
        detections_per_im, nms_thresh,
        scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());

    return {nms_scores, nms_boxes, nms_classes};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "SigmoidFocalLoss_forward");
  m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "SigmoidFocalLoss_backward");
  m.def("nv_decode", &nv_decode);
  m.def("nv_nms", &nv_nms);
}
