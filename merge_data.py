# 将 /root/DIS/DIS5K 下的 im 的图片与 gt 的图片合并到 /root/DIS/ay-item-data 下

import os
import shutil
import random

im_path = '/root/DIS/DIS5K/DIS-TR/im'
gt_path = '/root/DIS/DIS5K/DIS-TR/gt'

vd_path = '/root/DIS/DIS5K/DIS-VD/im'
vg_path = '/root/DIS/DIS5K/DIS-VD/gt'

ay_item_data_path_train_im = '/root/DIS/ay-item-data/train/im'
ay_item_data_path_train_gt = '/root/DIS/ay-item-data/train/gt'

ay_item_data_path_val_im = '/root/DIS/ay-item-data/val/im'
ay_item_data_path_val_gt = '/root/DIS/ay-item-data/val/gt'

ay_item_data_path_test_im = '/root/DIS/ay-item-data/test/im'
ay_item_data_path_test_gt = '/root/DIS/ay-item-data/test/gt'


# 随机挑选 xx 个图片 合并到 ay-item-data
def merge_data(im_path, gt_path, ay_item_data_path_im, ay_item_data_path_gt, num=100):
    im_list = os.listdir(im_path)
    
    # 打乱顺序
    random.shuffle(im_list)

    for i in range(num):
        im_name = im_list[i]
        # 同名但是不同后缀的图片
        gt_name = im_name.split('.')[0] + '.png'
        shutil.copy(os.path.join(im_path, im_name), os.path.join(ay_item_data_path_im, im_name))
        shutil.copy(os.path.join(gt_path, gt_name), os.path.join(ay_item_data_path_gt, gt_name))

merge_data(im_path, gt_path, ay_item_data_path_train_im, ay_item_data_path_train_gt, num=1320)
merge_data(vd_path, vg_path, ay_item_data_path_val_im, ay_item_data_path_val_gt, num=220)
merge_data(im_path, gt_path, ay_item_data_path_test_im, ay_item_data_path_test_gt, num=660)

