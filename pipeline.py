import numpy as np
from imgProcessing import *


# load image tif file into numpy array
with rasterio.open('../data/{}/{}.tif'.format(pplant_name, img_name)) as ds:
    img_arr = ds.read()


# crop image
img_arr = crop_image(img_arr, az_angle)

# Convert to greyscale
img_arr = convert_to_greyscale(img_arr)