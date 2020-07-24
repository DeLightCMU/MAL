from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import cv2

import os
import numpy as np
import random

cocoRoot = '/data/weik/MSCOCO2017/'
dataType = 'val2017'

annFile = os.path.join(cocoRoot, f'annotations/instances_{dataType}.json')
print(f'Annotation file: {annFile}')

# initialize COCO api for instance annotations
coco = COCO(annFile)

# 利用getCatIds函数获取某个类别对应的ID，
# 这个函数可以实现更复杂的功能，请参考官方文档
ids = coco.getCatIds('person')[0]
print(f'person id: {ids}')

# 利用loadCats获取序号对应的文字类别
# 这个函数可以实现更复杂的功能，请参考官方文档
cats = coco.loadCats(1)
print(f'1 class name: {cats}')

# 获取包含person的所有图片
imgIds = coco.getImgIds(catIds=[1])
print(f'images which include person are: {len(imgIds)}')

# 获取包含dog的所有图片
id = coco.getCatIds(['dog'])[0]
imgIds = coco.catToImgs[id]
print(f'images which include dog has: {len(imgIds)}')
print(imgIds)

# 展示图片信息
imgId = imgIds[10]

imgInfo = coco.loadImgs(imgId)[0]
print('imgInfo: ', imgInfo)

# imPath = os.path.join(cocoRoot, 'images', dataType, imgInfo['file_name'])
# im = cv2.imread(imPath)
# plt.axis('off')
# plt.imshow(im)
# plt.show()
# plt.imshow(im); plt.axis('off')

# 获取该图像对应的anns的Id
annIds = coco.getAnnIds(imgIds=imgInfo['id'])
anns = coco.loadAnns(annIds)
print(f'图像{imgInfo["id"]}包含{len(anns)}个ann对象，分别是:\n{annIds}')

coco.showAnns(anns)

print(f'ann{annIds[3]}对应的mask如下：')
mask = coco.annToMask(anns[3])
# plt.imshow(mask); plt.axis('off')