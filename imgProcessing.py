'''
README:
This python script performs image preprocessing, including:
	- image cropping
	- grey scale
	- shadow angle estimation

The below functions are supposed be called sequentially.
'''


import numpy as np


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
	grey_image = np.average(img, weights = weights, axis=2)
    return grey_image











