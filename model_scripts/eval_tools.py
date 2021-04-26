# Built-in
import os
import sys
import time

# Libs
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Libs
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams 

# Pytorch
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '../mrs/')

# Own modules
from data import data_loader, data_utils
from mrs_utils import misc_utils, vis_utils#, eval_utils
from network import network_io, network_utils

def make_colormap(n, zero_white=True):
    '''
    Creates a color map with n number of colors
    :param n: int, desired number of colors
    :param zero_white: if True, sets first color to white (255, 255, 255)
    :return: np.array with n colors (+1 if zero_black=True)
    '''
    import colorsys 
    def hsv2rgb(h, s, v):
        return [round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v)]
    
    colormap = np.zeros((n, 3))
    if zero_white:
        colormap[0] = np.array([255, 255, 255])
        for i, h in enumerate(np.arange(1 / (n - 1), n / (n - 1), 1 / (n - 1)) + 1 / 6):
            colormap[i + 1] = hsv2rgb(h, 1, 1)
        return colormap
    else:
        for i, h in enumerate(np.arange(1 / n, (n + 1) / n, 1 / n) + 1 / 6):
            colormap[i] = hsv2rgb(h, 1, 1)
        return colormap
    

def make_geo_file_list(data_dir, train_file, valid_file=None,
                       geographies=['barren', 'crops', 'developed', 'forest', 'grassland', 
                                   'open_water', 'pasture', 'scrub', 'wetlands']
                      ):
    '''
    Creates a file list for each type of geography
    :param data_dir: absolute path to patches directory
    :param train_file: absolute path to train_file
    :param valid_file: absolut path to valid_file, can be empty
    :param geographies: list of strings with names of geographies (present in file names)
    :return:
    '''
    import re
    if valid_file:
        files = [open(train_file, "r"), open(valid_file, "r")]
    else:
        files = [open(train_file, "r")]

    geo_files = {}
    for geo in geographies:
        geo_files[geo] = open(data_dir + '../' + geo + '.txt', 'w')


    for file in files:
        for line in file:
            line = line.strip()
            names = line.split()
            for geo in geographies:
                if re.match(r'^(\w+)_id', names[0]).group(1) == geo:
                    geo_files[geo].write(line + '\n')
                else:
                    pass
                
def get_predicted_images(test_loader, device, model, start_batch=0, num_batches=100):
    '''
    Creates 3 lists (images, labels, predictions) of batches of files (np.ndarrays) from the test_loader
    :param test_loader: DataLoader with images and labels (rasters)
    :param start_batch: first batch in DataLoader to be processed
    :param num_batches: number of batches in DataLoader to process
    :param device: device (as returned from mrs.mrs_utils.misc_utils.set_gpu() (1st output))
    :param model: model (as returned from mrs.network.network_io.create_model())
    :return img_images: list of batches (np.ndarrays) with images in test_loader
    :return lbl_images: list of batches (np.ndarrays) with labels (rasters) in test_loader
    :return pred_images: list of batches (np.ndarrays) with predictions from model
    '''
    img_images, lbl_images, pred_images = [], [], []
    
    for img_cnt, data_dict in enumerate(tqdm(test_loader)):
        if img_cnt < start_batch: continue
        if img_cnt == start_batch + num_batches: break
        image = Variable(data_dict['image'], requires_grad=True).to(device)
        label = Variable(data_dict['mask']).long().to(device)
        with torch.autograd.no_grad():
            output_dict = model.forward(image)
        img_images.append(image.detach().cpu().numpy())
        lbl_images.append(label.cpu().numpy())
        pred_images.append(output_dict['pred'].detach().cpu().numpy())
        
    return img_images, lbl_images, pred_images


def make_pictures(batch, i, img_images, lbl_images, pred_images, colormap, mean, std, exclude_zero_cls=False):
    '''
    Makes a list of pictures (RGB, raster, prediction, [prediction except class 0])
    :param batch: batch where desired image is located
    :param i: index within batch where desired image is located
    :param img_images: list of batches of images as returned by get_predicted_images()
    :param lbl_images: list of batches of labels as returned by get_predicted_images()
    :param pred_images: list of batches of predictions as returned by get_predicted_images()
    :param colormap: array of colors as returned by make_colormap()
    :param exlude_zero_cls: flag to indicate if a fourth image excluding class 0 should be plotted
    :param mean, std: params used to normalize original DataLoader
    :return pictures: list of desired pictures as np.arrays
    :return titles: list of strings with corresponding image titles
    '''
    img = vis_utils.inv_normalize(data_utils.change_channel_order(img_images[batch][i]), mean, std)
    lbl = colormap[lbl_images[batch][i]]
    pred = colormap[np.argmax(pred_images[batch][i][:], axis=0)]
    pictures = [img, lbl, pred]
    titles = ['RGB', 'Raster', 'Predicted']
    if exclude_zero_cls:
        pred2 = colormap[np.argmax(pred_images[batch][i][1:], axis=0)]
        titles.append('Predicted Except 0')
        pictures.append(pred2)
    return pictures, titles


def plot_results(batch, i, img_images, lbl_images, pred_images, colormap, mean, std, exclude_zero_cls=False):
    '''
    Plots pictures of RGB data, raster and prediction side by side
    :param batch: batch where desired image is located
    :param i: index within batch where desired image is located
    :param img_images: list of batches of images as returned by get_predicted_images()
    :param lbl_images: list of batches of labels as returned by get_predicted_images()
    :param pred_images: list of batches of predictions as returned by get_predicted_images()
    :param colormap: array of colors as returned by make_colormap()
    :param exlude_zero_cls: flag to indicate if a fourth image excluding class 0 should be plotted
    :param mean, std: params used to normalize original DataLoader
    :return:
    '''
    pictures, titles = make_pictures(batch, i, img_images, lbl_images, pred_images, 
                                     colormap, mean, std, exclude_zero_cls)
    
    from pylab import rcParams 
    rcParams['figure.figsize'] = 16, 5
    plt.figure()
    n_images = 4 if exclude_zero_cls else 3
    for j in range(n_images):
        plt.subplot(1, n_images, j + 1) 
        plt.imshow(pictures[j] / 255 if j != 0 else pictures[j]) 
        plt.title(titles[j]) 
        plt.xticks([])
        plt.yticks([])


def evaluate_class_predictions(lbl_images, pred_images, cls):
    '''
    Classifies pixels in pred_images as true positive (TP), false positive (FP),
    false negative (FN), or true negative (TN), then returns total count for each classification
    :param lbl_images: list of batches of labels as returned by get_predicted_images()
    :param pred_images: list of batches of predictions as returned by get_predicted_images()
    :param cls: desired class within images
    :return TP: number of true positives in lists
    :return FP: number of false positives in lists
    :return FN: number of false negatives in lists
    :return TN: number of true negatives in lists
    '''
    TP, FP, FN, TN = 0, 0, 0, 0

    for batch in tqdm(range(len(lbl_images))):
        for i in range(len(lbl_images[batch])):
            cls_image = np.equal(lbl_images[batch][i], cls) * 1
            pred_image = np.equal(np.argmax(pred_images[batch][i][:], axis=0), cls) * 1
            TP += np.sum(np.equal(cls_image + pred_image, 2))
            FP += np.sum(np.equal(cls_image - pred_image, -1))
            FN += np.sum(np.equal(cls_image - pred_image, 1))
            TN += np.sum(np.equal(cls_image + pred_image, 0))
    
    return TP, FP, FN, TN


def evaluate_class_predictions_step(cls, test_loader, device, model, batches_per_step=100):
    '''
    Classifies pixels in pred_images as true positive (TP), false positive (FP),
    false negative (FN), or true negative (TN), then returns total count for each classification.
    Use when test_loader is too big and should be processed every batches_per_step separately.
    :param cls: desired class within images
    :param test_loader: DataLoader with images and labels (rasters)
    :param device: device (as returned from mrs.mrs_utils.misc_utils.set_gpu() (1st output))
    :param model: model (as returned from mrs.network.network_io.create_model())
    :param batches_per_step: number of batches in DataLoader to process per step
    :return TP: number of true positives in test_loader
    :return FP: number of false positives in test_loader
    :return FN: number of false negatives in test_loader
    :return TN: number of true negatives in test_loader
    '''
    import time
    print('Loading first {} batches...'.format(batches_per_step))
    time.sleep(0.2)
    TP, FP, FN, TN = 0, 0, 0, 0

    for batch in range(0, len(test_loader), batches_per_step):
        _, lbl_images, pred_images = get_predicted_images(test_loader, device, model, batch, batches_per_step)
        print('Evaluating predictions...')
        time.sleep(0.2)
        cm_results = evaluate_class_predictions(lbl_images, pred_images, cls)
        TP += cm_results[0]
        FP += cm_results[1]
        FN += cm_results[2]
        TN += cm_results[3]
        if batch > len(test_loader) - batches_per_step: continue
        print('Loading next {} batches...'.format(batches_per_step))
        time.sleep(0.2)
    
    return TP, FP, FN, TN

def confusion_matrix(class_predictions, cls):
    '''
    Prints the confusion matrix for class cls with predictions in tuple class_predictions
    :param class_predictions: tuple with values (TP, FP, FN, TN)
    :param cls: class to be printed in confusion matrix
    :return metrics: (precision, npv, sensitivity, specificity, accuracy)
    :return totals: (actual_positives, actual_negatives, predicted_positives, predictive_negatives, total)
    '''
    cm = class_predictions
    TP, FP, FN, TN = cm[0], cm[1], cm[2], cm[3]

    precision = round(TP / (TP + FP), 4)
    npv = round(TN / (TN + FN), 4)
    sensitivity = round(TP / (TP + FN), 4)
    specificity = round(TN / (TN + FP), 4)

    accuracy = round((TP + TN) / sum(cm), 4)

    actual_positives = TP + FN
    actual_negatives = FP + TN
    predicted_positives = TP + FP
    predicted_negatives = FN + TN

    total = TP + FP + FN + TN

    cell_0 = f'Class {cls}'.ljust(15)

    cell_1 = 'Raster (Actual)'.ljust(15)
    cell_2 = f'Class {cls}'.ljust(15)
    cell_3 = f'{actual_positives:,}'.rjust(15)
    cell_4 = '{:.2f}%'.format(float(actual_positives / total * 100)).rjust(15)
    cell_5 = 'Other'.ljust(15)
    cell_6 = f'{actual_negatives:,}'.rjust(15)
    cell_7 = '{:.2f}%'.format(float(actual_negatives / total * 100)).rjust(15)

    cell_8 = 'Prediction'.ljust(15)
    cell_9 = f'Class {cls}'.ljust(15)
    cell_10 = f'{predicted_positives:,}'.rjust(15)
    cell_11 = '{:.2f}%'.format(float(predicted_positives / total * 100)).rjust(15)
    cell_12 = 'Other'.ljust(15)
    cell_13 = f'{predicted_negatives:,}'.rjust(15)
    cell_14 = '{:.2f}%'.format(float(predicted_negatives / total * 100)).rjust(15)

    cell_15 = f'{TP:,}'.rjust(15)
    cell_16 = '{:.2f}%'.format(float(TP / total * 100)).rjust(15)
    cell_17 = f'{FP:,}'.rjust(15)
    cell_18 = '{:.2f}%'.format(float(FP / total * 100)).rjust(15)
    cell_19 = f'{FN:,}'.rjust(15)
    cell_20 = '{:.2f}%'.format(float(FN / total * 100)).rjust(15)
    cell_21 = f'{TN:,}'.rjust(15)
    cell_22 = '{:.2f}%'.format(float(TN / total * 100)).rjust(15)

    cell_23 = 'Precision'.ljust(15)
    cell_24 = f'{precision}'.rjust(15)
    cell_25 = 'NPV'.ljust(15)
    cell_26 = f'{npv}'.rjust(15)
    cell_27 = 'Sensitivity'.ljust(15)
    cell_28 = f'{sensitivity}'.rjust(15)
    cell_29 = 'Specificity'.ljust(15)
    cell_30 = f'{specificity}'.rjust(15)
    cell_31 = 'Accuracy'.ljust(15)
    cell_32 = f'{accuracy}'.rjust(15)

    empty = ''.ljust(15)

    print('-' * 95)
    print(cell_0.ljust(47) + '|' + cell_1.ljust(31) + '|'.ljust(15))
    print(''.ljust(47) + '-' * 33 + ''.ljust(15))
    print(''.ljust(47) + '|' + '|'.join([cell_2, cell_5, empty]))
    print(''.ljust(47) + '-' * 33 + ''.ljust(15))
    print(''.ljust(47) + '|' + '|'.join([cell_3, cell_6, empty]))
    print(''.ljust(47) + '|' + '|'.join([cell_4, cell_7, empty]))
    print('-' * 95)
    print('|'.join([cell_8, cell_9, cell_10, cell_15, cell_17, cell_23]))
    print('|'.join([empty, empty, cell_11, cell_16, cell_18, cell_24]))
    print(' ' * 15 + '-' * 80)
    print('|'.join([empty, cell_12, cell_13, cell_19, cell_21, cell_25]))
    print('|'.join([empty, empty, cell_14, cell_20, cell_22, cell_26]))
    print('-' * 95)
    print(''.ljust(47) + '|' + '|'.join([cell_27, cell_29, cell_31]))
    print(''.ljust(47) + '|' + '|'.join([cell_28, cell_30, cell_32]))
    print('-' * 95)
    
    return ((precision, npv, sensitivity, specificity, accuracy), 
            (actual_positives, actual_negatives, 
             predicted_positives, predicted_negatives, total))

def plot_colormap(start_at, size, colormap, titles):
    '''
    Plots the colormap with titles for each class.
    :param start_at: first class in plot
    :param size: number of classes to plot
    :param colormap: np.array with RGB colors for colormap
    :param titles: dictionary with string class number as keys and string names as values
    :return:
    '''
    rcParams['figure.figsize'] = 0.5, size * 1.1
    plt.figure()
    for i in range(0 + start_at, size + start_at):
        plt.subplot(size, 1, i - start_at + 1) 
        plt.imshow([[colormap[i] / 255]]) 
        plt.title(str(i), pad=-10) 
        plt.xlabel(titles[str(i)])
        plt.xticks([])
        plt.yticks([])
        
def class_count(n_class, file_list, data_dir):
    '''
    Counts the number of pixels of each class in the rasters/labels in the file_list.
    :param n_class: int specifying number of classes
    :param file_list: list of strings with path to txt files with pairs of images and rasters
    :param data_dir: string with path to directory with raster files
    :return class_count: dictionary where keys are the class number and values are the total pixes
                         of that class
    '''
    class_count = {cls: 0 for cls in list(range(n_class))}

    for file in file_list:
        print('Processing {}...'.format(file))
        time.sleep(0.2)
        image_list = misc_utils.load_file(file)
        _, lbl_list = data_loader.get_file_paths(data_dir, image_list)
        for lbl_file in tqdm(lbl_list):
            lbl = misc_utils.load_file(lbl_file)
            for cls in list(range(n_class)):
                class_count[cls] += np.sum(np.equal(lbl, cls))
    return class_count