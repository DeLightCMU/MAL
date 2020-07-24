# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import os

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="./configs/mal_R-50-FPN_visdrone.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        # default=0.7,
        default=0.5,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    # im = cv2.imread(os.path.join('/data/weik/MSCOCO2017/images', '{0:012d}.jpg'.format(image_name)))
    image_name = '0000006_00611_d_0000002.jpg'
    # image_name = '000000159269.jpg'
    im = cv2.imread(os.path.join('/data3/brad/data/VisDrone2019/images', image_name))
    # im = cv2.imread(os.path.join('/data/weik/MSCOCO2017/images', image_name))
    composite = coco_demo.run_on_opencv_image(im)
    cv2.namedWindow('bbox', 0)
    cv2.resizeWindow('bbox', 640, 480)
    cv2.moveWindow('bbox', 0, 0)
    cv2.imshow("bbox", composite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('res.jpg', composite)

    # cam = cv2.VideoCapture(0)
    # while True:
    #     start_time = time.time()
    #     ret_val, img = cam.read()
    #     composite = coco_demo.run_on_opencv_image(img)
    #     print("Time: {:.2f} s / img".format(time.time() - start_time))
    #     cv2.imshow("COCO detections", composite)
    #     if cv2.waitKey(1) == 27:
    #         break  # esc to quit
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()