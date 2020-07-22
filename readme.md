# Multiple Anchor Learning (MAL)

This is the official implementation of the paper: 
- Wei Ke, Tianliang Zhang, Zeyi Huang, Qixiang Ye, Jianzhuang Liu, Dong Huang, Multiple Anchor Learning for Visual Object Detection. [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ke_Multiple_Anchor_Learning_for_Visual_Object_Detection_CVPR_2020_paper.pdf), CVPR, 2020. 

This repo includes the basic training and inference pipeline based on [maskrcnn_benckmark](https://github.com/facebookresearch/maskrcnn-benchmark) . 

For fast inference, please direct to [MAL-inference]( https://github.com/DeLightCMU/MAL-inference)


## 1. Installation

### Requirements:
- Python3
- PyTorch 1.1 with CUDA support
- torchvision 0.2.1
- pycocotools
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name free_anchor python=3.7
conda activate free_anchor

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm

# pytorch and torchvision
# we give the instructions for CUDA 10.0
conda install pytorch=1.1 torchvision=0.2.1 cudatoolkit=10.0 -c pytorch

# install pycocotools
pip install pycocotools

# install FreeAnchor
git clone https://github.com/DeLightCMU/MAL.git

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
cd MAL
python setup.py build develop

# or if you are on macOS
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```


## 2. Running

For that, all you need to do is to modify `maskrcnn_benchmark/config/paths_catalog.py` to point to the location where your dataset is stored.

#### Config Files
We provide four configuration files in the configs directory.

| Config File              | Backbone                | Iteration |
| :----------------------: | :---------------------: | :-------: |
| configs/mal_R-50-FPN     | ResNet-50-FPN           | 135k      |
| configs/mal_R-101-FPN    | ResNet-101-FPN          | 180k       |
| configs/mal_X-101-FPN    | ResNext-101-FPN         | 180k      |


#### Training with 8 GPUs

```bash
cd path_to_mal
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "path/to/config/file.yaml"
```

#### Test on COCO test-dev

```bash
cd path_to_free_anchor
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "path/to/config/file.yaml" MODEL.WEIGHT "path/to/.pth file" DATASETS.TEST "('coco_test-dev',)"
```

#### Multi-scale testing

```bash
cd path_to_free_anchor
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/multi_scale_test.py --config-file "path/to/config/file.yaml" MODEL.WEIGHT "path/to/.pth file" DATASETS.TEST "('coco_test-dev',)"
```


### 3. Model
