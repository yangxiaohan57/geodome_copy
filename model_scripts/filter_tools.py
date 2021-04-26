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


def get_features(file_list_paths, data_dir, feature_extractor=None):
    labels = list(range(len(file_list_paths)))

    if feature_extractor is None:
        X = np.empty((0, 8))
        y = np.empty((0,))
        file_names = np.empty((0), dtype='object')

        for l in labels:
            file_list = misc_utils.load_file(file_list_paths[l])
            img_list, lbl_list = data_loader.get_file_paths(data_dir, file_list)
            files = zip(img_list, lbl_list)

            X_temp = np.zeros((len(file_list), 8))
            y_temp = np.zeros((len(file_list),))

            for i, (img_file, lbl_file) in enumerate(tqdm(zip(img_list, lbl_list))):
                img = misc_utils.load_file(img_file)
                lbl = misc_utils.load_file(lbl_file)
                X_temp[i, 0:3] = np.mean(img, axis=(0, 1))
                X_temp[i, 3:6] = np.std(img, axis=(0, 1))
                X_temp[i, 6] = np.mean(lbl)
                X_temp[i, 7] = np.std(lbl)
                y_temp[i] = float(l)
                file_names = np.append(file_names, img_file)

            X = np.append(X, X_temp, axis=0)
            y = np.append(y, y_temp, axis=0)

        shuffle_indexes = np.arange(X.shape[0])
        np.random.shuffle(shuffle_indexes)

        X = X[shuffle_indexes]
        y = y[shuffle_indexes]
        file_names = file_names[shuffle_indexes]

        return X, y, file_names

def filter_data(data_dir, file_list_path, save_dir, save_file_name, valid_geo=None,
                min_n_classes=1, min_class_mean=0, min_class_std=0, classifier=None,
                exclude_list_path=None):
    misc_utils.make_dir_if_not_exist(save_dir)
    file_list = misc_utils.load_file(file_list_path)
    img_list, lbl_list = data_loader.get_file_paths(data_dir, file_list)
    exclude_list = []

    if exclude_list_path:
        exclude_list = misc_utils.load_file(exclude_list_path)

    record_file_train = open(os.path.join(save_dir, save_file_name + '_train.txt'), 'w+')
    if valid_geo:
        record_file_valid = open(os.path.join(save_dir, save_file_name + '_valid.txt'), 'w+')

    files = zip(img_list, lbl_list)
    for x, y in files:
        assert x[:-4] == y[:-11]
    #pdb.set_trace()
    for cnt, (img_file, lbl_file) in enumerate(tqdm(zip(img_list, lbl_list))):
        if os.path.exists(img_file[:-4] + '_N.jpg'):
            os.remove(img_file[:-4] + '_N.jpg')
        img = misc_utils.load_file(img_file)
        lbl = misc_utils.load_file(lbl_file)
        n_classes = np.unique(lbl).shape[0]
        class_mean = np.mean(lbl)
        class_std = np.std(lbl)
        y_hat = 1

        if classifier:
            X = np.zeros((1, 8))
            X[0, 0:3] = np.mean(img, axis=(0, 1))
            X[0, 3:6] = np.std(img, axis=(0, 1))
            X[0, 6] = np.mean(lbl)
            X[0, 7] = np.std(lbl)
            y_hat = classifier.predict(X)[0]

        if ((n_classes >= min_n_classes) and (class_mean >= min_class_mean)
            and (class_std >= min_class_std) and (y_hat == 1)
            and (file_list[cnt] not in exclude_list)):
            img_name = file_list[cnt].split()[0]
            lbl_name = file_list[cnt].split()[1]

            if re.match(r'^(\w+)_id', img_name).group(1) == valid_geo:
                record_file_valid.write('{} {}\n'.format(img_name, lbl_name))
            else:
                record_file_train.write('{} {}\n'.format(img_name, lbl_name))

    record_file_train.close()
    if valid_geo:
        record_file_valid.close()
