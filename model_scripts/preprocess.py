# Built-in
import os
import re

# Libs
import numpy as np
from tqdm import tqdm
import sys
from natsort import natsorted

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '../mrs/')

from data import data_loader, data_utils
from mrs_utils import misc_utils, process_block

# Settings
DS_NAME = 'Geodome'
MEAN = (0.40994515, 0.38314009, 0.28864455)
STD = (0.12889884, 0.10563929, 0.09726452)
RGB_DIR = '/home/nas-mount/550_stacked_bands/'
GT_DIR = '/home/jlm206/rasterized_tags'
GEOS = ['barren', 'crops', 'developed', 'forest', 'grassland', 'open_water', 'pasture', 'scrub', 'wetlands']

def get_file_list(data_dir):
    _, _, files = next(os.walk(data_dir, topdown=True))
    for i, file_name in enumerate(files):
        if file_name[-4:] != '.png':
            files.pop(i)
    return natsorted(files)

def get_images(rgb_dir, gt_dir):
    rgb_list = get_file_list(rgb_dir)
    gt_list = get_file_list(gt_dir)
    rgb_files, gt_files = [], []
    for file_name in gt_list:
        file_id = file_name[:-11]
        if (file_id + '.png' in rgb_list) and (file_id + '_N.png' in rgb_list):
            rgb_files.append(file_id + '.png')
            gt_files.append(file_name)
    return rgb_files, gt_files

def patch_tile(rgb_file, gt_file, rgb_dir, gt_dir, patch_size=[500, 500], n_patches=2):
    """
    Extract the given rgb and gt tiles into patches
    :param rgb_file: path to the rgb file
    :param gt_file: path to the gt file
    :param patch_size: size of the patches, should be a tuple of (h, w)
    :return: rgb and gt patches as well as coordinates
    """
    rgb = misc_utils.load_file(os.path.join(rgb_dir, rgb_file))
    n = misc_utils.load_file(os.path.join(rgb_dir, rgb_file[:-4] + '_N.png'))
    gt = misc_utils.load_file(os.path.join(gt_dir, gt_file))
    np.testing.assert_array_equal(rgb.shape[:2], gt.shape)

    # Making image at least the size of the patch_size
    if gt.shape[0] < patch_size[0]:
        gt = np.append(gt, np.zeros((patch_size[0] - gt.shape[0], gt.shape[1]), dtype='uint8'), axis=0)
        n = np.append(n, np.zeros((patch_size[0] - n.shape[0], n.shape[1]), dtype='uint8'), axis=0)
        rgb = np.append(rgb, np.zeros((patch_size[0] - rgb.shape[0],
                                       rgb.shape[1], rgb.shape[2]), dtype='uint8'), axis=0)
    if gt.shape[1] < patch_size[1]:
        gt = np.append(gt, np.zeros((gt.shape[0], patch_size[1] - gt.shape[1]), dtype='uint8'), axis=1)
        n = np.append(n, np.zeros((n.shape[0], patch_size[1] - n.shape[1]), dtype='uint8'), axis=1)
        rgb = np.append(rgb, np.zeros((rgb.shape[0], patch_size[1] - rgb.shape[1],
                                       rgb.shape[2]), dtype='uint8'), axis=1)

    y = [0, gt.shape[0] - patch_size[0], 0, gt.shape[0] - patch_size[0]]
    x = [0, gt.shape[1] - patch_size[1], gt.shape[1] - patch_size[1], 0]
    suf = ['NW', 'SE', 'NE', 'SW']

    for i in range(n_patches):
        rgb_patch = data_utils.crop_image(rgb, y[i], x[i], patch_size[0], patch_size[1])
        n_patch = data_utils.crop_image(n, y[i], x[i], patch_size[0], patch_size[1])
        gt_patch = data_utils.crop_image(gt, y[i], x[i], patch_size[0], patch_size[1])

        yield rgb_patch, gt_patch, n_patch, suf[i]

def patch_geodome(rgb_files, gt_files, rgb_dir, gt_dir, save_dir, patch_size=[500, 500],
                  n_patches=2, valid_geo='barren', N_channel=False):
    # create folders and files
    patch_dir = os.path.join(save_dir, 'patches')
    misc_utils.make_dir_if_not_exist(patch_dir)
    record_file_train = open(os.path.join(save_dir, 'file_list_train.txt'), 'w+')
    record_file_valid = open(os.path.join(save_dir, 'file_list_valid.txt'), 'w+')
    files = zip(rgb_files, gt_files)
    for x, y in files:
        assert x[:-4] == y[:-11]

    for cnt, (img_file, lbl_file) in enumerate(tqdm(zip(rgb_files, gt_files))):
        for rgb_patch, gt_patch, n_patch, suf in patch_tile(img_file, lbl_file, rgb_dir, gt_dir, patch_size, n_patches):
            img_patchname = img_file[:-4] + '_' + suf + '.jpg'
            lbl_patchname = lbl_file[:-10] + suf + '_raster.png'
            n_patchname = img_file[:-4] + '_' + suf + '_N.jpg'

            misc_utils.save_file(os.path.join(patch_dir, img_patchname), rgb_patch.astype(np.uint8))
            misc_utils.save_file(os.path.join(patch_dir, lbl_patchname), gt_patch.astype(np.uint8))
            if N_channel:
                misc_utils.save_file(os.path.join(patch_dir, n_patchname), n_patch.astype(np.uint8))

            if re.match(r'^(\w+)_id', img_file).group(1) == valid_geo:
                record_file_valid.write('{} {}\n'.format(img_patchname, lbl_patchname))
            else:
                record_file_train.write('{} {}\n'.format(img_patchname, lbl_patchname))
    record_file_train.close()
    record_file_valid.close()

def get_stats(img_dir):
    from data import data_utils
    rgb_imgs = [a[0] for a in data_utils.get_img_lbl(img_dir, '.jpg', 'raster.png')]
    ds_mean, ds_std = data_utils.get_ds_stats(rgb_imgs)
    return np.stack([ds_mean, ds_std], axis=0)

if __name__ == '__main__':
    ps = 512
    valid_geo=None
    save_dir = r'/home/ss1072/data_210402-test'
    misc_utils.make_dir_if_not_exist(save_dir)
    rgb_files, gt_files = get_images(RGB_DIR, GT_DIR)
    patch_geodome(rgb_files=rgb_files,
                  gt_files=gt_files,
                  rgb_dir=RGB_DIR,
                  gt_dir=GT_DIR,
                  save_dir=save_dir,
                  patch_size=[ps, ps],
                  n_patches=2,
                  valid_geo=valid_geo,
                  N_channel=False)
