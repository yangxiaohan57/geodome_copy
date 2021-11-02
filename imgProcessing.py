##############################################################
# README:
# This python script performs image preprocessing, including:
# 	- image cropping
#   - grey scale
#	- shadow angle estimation
#
# The below functions are supposed be called sequentially.
##############################################################

import numpy as np
import rasterio
import pickle

'''
Get all image names function
- This function gets all the images names downloaded and stored in "/data/powerplantname"

:param
'''
def get_all_img_names(meta_data, plant_name):
	# read in metadata as a dict, with image names as keys and each image's metadata as values
	return meta_data[plant_name].keys()



'''
Perform band stacking function
- it calls func: stack_bands iteratively and stacks the bands for every images.

:param img_name_lst: list of all image names
'''
def perform_band_stacking(img_name_lst, dir):
	for img in img_name_lst:
		stack_bands(dir + '/' + str(img), dir + '/' + str(img) + '.tif')



'''
Based on Kyle's code
(!!!!!!!!Private function, shoudn't be called from outside this script!!!!!!!!!!)

Stack bands function
- This function stacks the RGB bands tif file to become a visible image.

:params img_name: the name of the 3 bands' tif files
:params output_name: the name of the stacked output
'''
def stack_bands(img_name, output_name):
	file_list = [img_name + '.B4.tif', img_name + '.B3.tif', img_name + '.B2.tif']

	# Read metadata of first file
	with rasterio.open(file_list[0]) as img0:
		meta = img0.meta
	# Update meta to reflect the number of layers
	meta.update(count = len(file_list))

	# Read each layer and write it to stack
	with rasterio.open(output_name, 'w', **meta) as dst:
		for id, layer in enumerate(file_list, start=1):
			with rasterio.open(layer) as src1:
				dst.write_band(id, src1.read(1))



#################################################################################
### 						Image pre-processing below                        ###
#################################################################################


'''
Note: Prior to calling this function, the image tif file should be load in as Numpy array using rastorio

Image cropping function:
- This function uses the angle of the shadow to determine which direction to crop the image.

:params image_arr: the numpy array of the original non-cropped image
:params az_angle: solar azmuth angle used to determine the angle of the shadow
:params crop_frac: the fraction of how much the cropped images takes up in the original image
'''
def crop_image(image_arr, az_angle, crop_frac=0.1):
	img_height = image_arr.shape[0]
	img_width = image_arr.shape[1]

	# keep the bottom left of the image
	if az_angle < 90:
		return image_arr[int(img_height/2) : int(img_height*(1-crop_frac)), int(img_width*crop_frac) : int(img_width/2)]
	# keep top left of the image
	elif az_angle < 180:
		return image_arr[int(img_height*crop_frac) : int(img_height/2), int(img_width*crop_frac):int(img_width/2)]
	# keep top left of the image
	elif az_angle < 270:
		return image_arr[int(img_height*crop_frac) : int(img_height/2), int(img_width/2):int(img_width*(1-crop_frac))]

    # keep top left of the image
	else:
		return image_arr[int(img_height/2) : int(img_height*(1-crop_frac)), int(img_width/2):int(img_width*(1-crop_frac))]



'''
Converting to greyscale function:
- This function reads in an image's numpy array and convert the image into greyscale using the weights

:params image_arr: the numpy array of the image to convert
:params weights: the weights applied to BGR bands. Set to [0.2989, 0.5870, 0.1140] by default
'''
def convert_to_greyscale(image_arr, weights=[0.2989, 0.5870, 0.1140]):
	grey_image = np.average(image_arr, weights = weights, axis=2)
	return grey_image
