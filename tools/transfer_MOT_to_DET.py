import json
import os
import cv2
import shutil

# phase = 'train'
phase = 'val'

g_sample_step = 10

root_path = '/data3/brad/data/VisDrone2019-MOT/'
generate_dataset_path = '/home/brad/VisDroneMotDataset_' + str(g_sample_step) + '/'

#datasets
anno_file_path  = os.path.join(root_path, 'VisDrone2019-MOT-' + phase + '/annotations')
image_file_path = os.path.join(root_path, 'VisDrone2019-MOT-' + phase + '/sequences')

# Annotations
class C_ANNO_INFO:
    def __init__(self):
        self.frame_index = 0
        self.target_id   = 0
        self.bbox_left   = 0
        self.bbox_top    = 0
        self.bbox_width  = 0
        self.bbox_height = 0
        self.score       = 0
        self.obj_cat     = 0
        self.trunc       = 0
        self.occlusion   = 0
        self.full_img_name  = ''
        self.full_anno_name = ''

def Cmp(input):
    return input.frame_index

def show_anno_info(anno):
    for i in range(len(anno)):
        print(anno[i].frame_index, anno[i].target_id, anno[i].bbox_left, anno[i].bbox_top, anno[i].bbox_width, anno[i].bbox_height,
              anno[i].score, anno[i].obj_cat, anno[i].trunc, anno[i].occlusion)

def collect_anno_info(sample_step=10):
    anno_file_list = sorted(os.listdir(anno_file_path))
    anno_names = [f for f in anno_file_list]
    print(phase + '_anno_names: ', anno_names)
    for f in anno_file_list:
        print('f: ', f)
        with open(anno_file_path + '/' + f) as fread:
            anno_lines = fread.readlines()

            anno_list = []
            for i, anno_info in enumerate(anno_lines):
                anno = C_ANNO_INFO()
                anno_info = anno_info.strip().split(',')
                anno.frame_index = int(anno_info[0])
                anno.target_id   = int(anno_info[1])
                anno.bbox_left   = int(anno_info[2])
                anno.bbox_top    = int(anno_info[3])
                anno.bbox_width  = int(anno_info[4])
                anno.bbox_height = int(anno_info[5])
                anno.score       = int(anno_info[6])
                anno.obj_cat     = int(anno_info[7])
                anno.trunc       = int(anno_info[8])
                anno.occlusion   = int(anno_info[9])

                img_name = "%07d" % int(anno_info[0])
                full_img_name = f[:-4] + '_' + img_name + '.jpg'
                anno.full_img_name = full_img_name

                anno_name = "%07d" % int(anno_info[0])
                full_anno_name = f[:-4] + '_' + anno_name + '.txt'
                anno.full_anno_name = full_anno_name

                anno_list.append(anno)

            anno_list.sort(key=Cmp)

            iter_index = 0
            last_full_anno_name = anno_list[iter_index].full_anno_name
            while iter_index < len(anno_list):
            # for iter_index in range(0, len(anno_list), sample_step):
                dir_name = generate_dataset_path + 'VisDrone2019_' + phase + '_annotation/'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                # print(iter_index, len(anno_list), anno_list[iter_index].full_anno_name)

                # ignore sample_step times
                for ignore_idx in range(sample_step):
                    while iter_index < len(anno_list):
                        if last_full_anno_name != anno_list[iter_index].full_anno_name:
                            break
                        iter_index = iter_index + 1
                    if iter_index < len(anno_list):
                        last_full_anno_name = anno_list[iter_index].full_anno_name
                # ignore sample_step times

                # print(iter_index, sample_step, len(anno_list))
                if iter_index < len(anno_list):
                    with open(dir_name + anno_list[iter_index].full_anno_name, 'a+') as fwrite:
                        while iter_index < len(anno_list):
                            if last_full_anno_name == anno_list[iter_index].full_anno_name:
                                fwrite.writelines(str(anno_list[iter_index].bbox_left) + ',' +
                                                  str(anno_list[iter_index].bbox_top) + ',' +
                                                  str(anno_list[iter_index].bbox_width) + ',' +
                                                  str(anno_list[iter_index].bbox_height) + ',' +
                                                  str(anno_list[iter_index].score) + ',' +
                                                  str(anno_list[iter_index].obj_cat) + ',' +
                                                  str(anno_list[iter_index].trunc) + ',' +
                                                  str(anno_list[iter_index].occlusion))
                                fwrite.writelines('\n')
                            else:
                                last_full_anno_name = anno_list[iter_index].full_anno_name
                                iter_index = iter_index + 1
                                break
                            iter_index = iter_index + 1

                #     if iter_index < len(anno_list):
                #         last_full_anno_name = anno_list[iter_index].full_anno_name
                # if iter_index < len(anno_list):
                #     print('write ', dir_name + anno_list[iter_index].full_anno_name)


# Images
# def collect_images():
#     image_file_list = sorted(os.listdir(image_file_path))
#     image_names = [f for f in image_file_list]
#     print('image_names: ', image_names)
#     for f in image_file_list:
#         f_path = os.path.join(root_path, 'VisDrone2019-MOT-' + phase + '/sequences/' + f)
#         sub_file_list = sorted(os.listdir(f_path))
#         for sub_f in sub_file_list:
#             new_image_name = f + '_' + sub_f
#             print('new_image_name: ', new_image_name)
#
#             src_path = f_path + '/' + sub_f
#             dst_path = generate_dataset_path + '/VisDrone2019_' + phase + '_images/'
#             if not os.path.exists(dst_path):
#                 os.makedirs(dst_path)
#
#             shutil.copy(src_path, dst_path + new_image_name)

# Images
def collect_images():
    anno_file_list = sorted(os.listdir(generate_dataset_path + 'VisDrone2019_' + phase + '_annotation/'))
    for f in anno_file_list:
        sep_names = f.split('_v_')
        image_folder = root_path + 'VisDrone2019-MOT-' + phase + '/sequences/' + sep_names[0] + '_v/'
        image_name   = sep_names[1][:-4] + '.jpg'
        src_path = image_folder + image_name
        dst_path = generate_dataset_path + '/VisDrone2019_' + phase + '_images/'

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        shutil.copy(src_path, dst_path + f[:-4] + '.jpg')


def main():
    # train dataset
    collect_anno_info(sample_step=g_sample_step)
    collect_images()


if __name__ == "__main__":
    main()