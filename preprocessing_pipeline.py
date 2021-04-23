"""
CLI tool to merge RGB bands and store meta of tiff files
"""

import os
import pickle
from argparse import ArgumentParser
from glob import glob

import numpy as np
import rasterio
from skimage.io import imread, imsave
from tqdm import tqdm

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def stack_RGB_bands(directory):
    img = np.stack(
        [
            imread("{}/download.R.tif".format(directory)),
            imread("{}/download.G.tif".format(directory)),
            imread("{}/download.B.tif".format(directory)),
        ],
        -1,
    )
    return img


def read_tiff_meta(geotiff_path):
    """Read geotiff, return reshaped image and metadata."""
    with rasterio.open(geotiff_path, "r") as src:
        img_meta = src.meta
        pixel_size_x, pixel_size_y = src.res
    return img_meta, pixel_size_x, pixel_size_y


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--in_dir", help="path to data directory", type=str)
    parser.add_argument(
        "-o",
        "--out_dir",
        help="path to processed data directory",
        default="./processed_data",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--meta_file",
        help="path to file containing meta information about the data",
        default="processed_data/meta.pkl",
        type=str,
    )

    args = parser.parse_args()

    meta_dict = {}

    img_dirs = glob("{}/*".format(args.in_dir))

    if not os.path.exists(args.out_dir):
        print("making output directory")
        os.makedirs(args.out_dir)
        pass

    # expected data directory structure is a directory for each image (as is the default output from the download pipeline)
    for directory in tqdm(img_dirs, desc="processing images"):
        img = stack_RGB_bands(directory)  # dont stack infrared band

        # the name of the directory has land cover type and point ID, we use it to name the file
        filename = directory.split("/")[-1]  # the split char could be different if
        imsave("{}/{}.png".format(args.out_dir, filename), img)

        # if infrared band exists save separately
        try:
            imsave(
                "{}/{}_N.png".format(args.out_dir, filename),
                imread("{}/download.N.tif".format(directory)),
            )
        except:
            pass

        # add meta information to the dictionary
        meta_dict[filename] = read_tiff_meta("{}/download.R.tif".format(directory))
        pass

    with open(args.meta_file, "wb") as f:
        pickle.dump(meta_dict, f)
        pass
    pass
