import sys
import numpy as np
import pandas as pd
from segment import plot_threshold
from segment import threshold
from skimage.filters import threshold_mean
import matplotlib.pyplot as plt
from imgProcessing import *
from helper import *
from height_estimation import *
from segment import *

# Load Metadata
with open("data/meta", "rb") as f:
    meta_data = pickle.load(f)

# Get all get_plant_names
plant_names = meta_data.keys()

data = []
for pplant_name in plant_names:
    # Loop through the images in each plant
    img_names = get_all_img_names(meta_data, pplant_name)
    for image_name in img_names:
        # Extract date
        date = image_name[:8]
        # mean azimuth angle
        az_angle = meta_data['{}'.format(pplant_name)]['{}'.format(image_name)]['properties']['MEAN_SOLAR_AZIMUTH_ANGLE']
        # mean zenith angle
        zn_angle = meta_data['{}'.format(pplant_name)]['{}'.format(image_name)]['properties']['MEAN_SOLAR_ZENITH_ANGLE']
        # cloud coverage
        coverage = meta_data['{}'.format(pplant_name)]['{}'.format(image_name)]['properties']['CLOUDY_PIXEL_PERCENTAGE']

        # Convert image to numpy array
        with rasterio.open('{}/{}.tif'.format(pplant_name, image_name)) as ds:
            img_arr = ds.read()  # read all raster values


        #Image Processing
        img_arr = np.moveaxis(img_arr, 0, -1)/4095
        cropped = crop_image(img_arr, az_angle)
        gray_cropped = convert_to_greyscale(cropped)

        # Caluclation of slope & intercept
        intercept, slope = linear_graph(gray_cropped, az_angle)


        ###################
        # threshold image #
        ###################

        # calculate threshold - use mean metric
        thresh = threshold_mean(gray_cropped)
        # create binary image
        img_arr = threshold(gray_cropped, thresh)

        # save thresholded image
        dir = '{}/{}.png'.format(pplant_name, image_name)
        plot_threshold(img_arr, slope, dir)

        # Connected connected
        labeled, nr_objects = connected_components(img_arr)
        xy = xy_coordinates(labeled)
        intersect = intersection_pt(intercept, slope, xy, labeled)
        dir = '{}/{}.png'.format(pplant_name, image_name)
        plot_cc(img_arr, labeled, slope, intersect, dir)
        shadow_len = dist(intersect, img_arr, slope)
        stack_height = calculate_stackheight(shadow_len, zn_angle, slope)
        data.append([image_name, pplant_name, date, coverage, zn_angle, az_angle, nr_objects, slope, intercept, shadow_len, stack_height])
df = pd.DataFrame(data, columns = ['image_name', 'powerplant', 'date', 'cloud_coverage', 'zenith_angle', 'azimuth_angle', 'blobs', 'slope', 'intercept', 'shadow_length', 'stack_height'])
df.to_csv('data/results.csv', index = False)
