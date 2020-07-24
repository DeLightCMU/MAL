import json
import os
import cv2

# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
# root_path = '/data3/brad/data/VisDrone2019-DET/'
#root_path = '/home/jimuyang/Workspace3/AIDTR/nrec_drone-val/'
root_path = './'

# 用于创建训练集或验证集
phase = 'train'
#phase = 'val'
#phase = 'test-dev'

# 训练集和验证集划分的界线
split = 20000

# 打开类别标签
classes = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']

# 建立类别标签和数字id的对应关系
dataset = {'categories':[], 'images':[], 'annotations':[]}
for i, cls in enumerate(classes):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

# 读取images文件夹的图片名称
# indexes = [f for f in os.listdir(os.path.join(root_path, 'images'))]
# 读取train2017文件夹的图片名称
# 改了，加了sorted，保证图片按名字升序，使得<'image_id': k >里面，k=0,对应第一张图片，00...01.jpg
# file_path = os.path.join(root_path, 'images')
file_path = os.path.join(root_path, 'annotations')
file_list = sorted(os.listdir(file_path))
_names = [f for f in file_list]

# 判断是建立训练集还是验证集
# if phase == 'train':
#     names = [line for i, line in enumerate(_names) if i <= split]
# elif phase == 'val':
#     names = [line for i, line in enumerate(_names) if i > split]
names = [line for i, line in enumerate(_names)]
# print('names: ', names)

# 读取Bbox信息
# 先将图片对应的标注文件的路径得到
_anno_names = [line[:-4]+'.txt' for i, line in enumerate(_names)]
# print('_anno_names: ', _anno_names)

annos = []
for i, anno_fn in enumerate(_anno_names):
    with open(os.path.join(root_path, 'annotations/'+anno_fn)) as tr:
        anno_list = tr.readlines()
        # print('debug: ', anno_fn, anno_list)
        # anno = {'filename':None, 'x_min':None, 'y_min':None, 'x_max': None, 'y_max': None, 'label': None}
        for j, _anno_list in enumerate(anno_list):
            anno = {}

            _anno_list = _anno_list.strip().split(',')
            # print('_anno_list: ', _anno_list)
            anno['index'] = anno_fn[:-4]
            anno['filename'] = anno_fn[:-4] + '.jpg'
            anno['x_min']  = int(_anno_list[0])
            anno['y_min']  = int(_anno_list[1])
            anno['width']  = int(_anno_list[2])
            anno['height'] = int(_anno_list[3])
            # print('width and height: ', anno['width'], anno['height'])
            anno['x_max'] = anno['x_min'] + anno['width']
            anno['y_max'] = anno['y_min'] + anno['height']
            # print('x2 and y2: ', anno['x_min'], anno['y_min'], anno['width'], anno['height'], anno['x_max'], anno['y_max'])
            anno['label'] = int(_anno_list[5])

            #
            if anno['label'] > 0:
                anno['label'] -= 1
                annos.append(anno)
            # if 0 == anno['label']:
            #     anno['label'] = 1
            # print('_anno_list[5]: ', _anno_list[5])
            # print('annos: ', annos)
            # annos.append(anno)
# print('annos: ', annos)
print('Total images are : ', len(names))

debug_vis = False

image_file_path = os.path.join(root_path, 'images')
image_file_list = sorted(os.listdir(image_file_path))
_image_names = [f for f in image_file_list]
image_names = [line for i, line in enumerate(_image_names)]

# 以上数据转换为COCO所需要的
for k, name in enumerate(image_names, 1):
    # if k > 3:
    #     break

    # 用opencv读取图片，得到图像的宽和高
    # print('name: ', image_file_path, name)
    im = cv2.imread(os.path.join(image_file_path, name))
    height, width, _ = im.shape

    # 添加图像的信息到dataset中
    dataset['images'].append({'file_name': name,
                              'id': k,
                              'width': width,
                              'height': height})

    # index = str(int(name[:-4]))
    index = str(name[:-4])
    # 一张图多个框时需要判断
    bFlag = False
    bboxes = []
    segs   = []
    for ii, anno in enumerate(annos, 1):
        # if index == anno['index'] and anno['label'] != 0:
        if index == anno['index']:
            bFlag = True
            # 类别
            cls_id = anno['label']
            # print('cls_id: ', cls_id)
            # x_min
            x1 = float(anno['x_min'])
            # y_min
            y1 = float(anno['y_min'])
            width = float(anno['width'])
            height = float(anno['height'])
            x2 = float(anno['x_max'])
            y2 = float(anno['y_max'])
            # print(cls_id, x1, y1, x2, y2, width, height)

            bbox = [x1, y1, width, height]
            seg  = [x1, y1, x2, y1, x2, y2, x1, y2]

            # bboxes.append(bbox)
            # segs.append(seg)
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': int(cls_id),
                'id': ii,
                'image_id': k,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })

            if True == debug_vis:
                print('x1, y1, x2, y2: ', x1, y1, x2, y2)
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
    if False == bFlag:
        #print('bug or there is no corresponding anno files for images (MOT dataset sampled with sample_step.')
        print('bug or there is no corresponding anno files for images:', name)

    # print('len(bboxes): ', len(bboxes))
    # dataset['annotations'].append({
    #     'area': width * height,
    #     'bbox': bboxes,
    #     'category_id': int(cls_id),
    #     'id': ii,
    #     'image_id': k,
    #     'iscrowd': 0,
    #     # mask, 矩形是从左上角点按顺时针的四个顶点
    #     'segmentation': segs
    # })

    # Debug
    if True == debug_vis:
        cv2.namedWindow('bbox', 0)
        cv2.resizeWindow('bbox', 640, 480)
        cv2.moveWindow('bbox', 0, 0)
        cv2.imshow('bbox', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print('len(dataset[annotations]): ', len(dataset['annotations']))

# 保存结果
#folder = '/data3/brad/data/VisDrone2019-DET'
folder = './'

if not os.path.exists(folder):
    os.makedirs(folder)
json_name = os.path.join(folder, '{}.json'.format(phase))
with open(json_name, 'w') as f:
    json.dump(dataset, f)
print('done')
