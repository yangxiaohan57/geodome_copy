"""
CLI tool to download satellite imagery from google earth engine
"""

import ee
import numpy as np
import pandas as pd
import pickle
import os
import rasterio
from osm_tools import _bbox_from_point, gdf_from_bbox

ee.Initialize()

from skimage.io import imread

from geetools import batch
from tqdm import tqdm

from argparse import ArgumentParser


def osm2gee_bbox(bbox):
    """
    convert bbox from lat lon (OSM default) to lon lat (GEE default)
    """
    return bbox[1], bbox[0], bbox[3], bbox[2]


def download_sen2_toLocal(bbox, name, meta_dict):
    """
    downloads NAIP imagery from the specified bounding box and saves it as `name`
    """
    AOI = ee.Geometry.Rectangle(list(bbox), "EPSG:4326", False)

    start_date = "2020-01-01"
    end_date = "2020-06-30"

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
        .filterBounds(AOI)
        .select(['B2', 'B3', 'B4'])
    )
    # Original code from previous year's:
    # image = ee.Image(collection.mosaic()).clip(AOI)
    # batch.image.toLocal(image, name, scale=scale, region=AOI)

    export_image(collection, name, meta_dict, region=AOI)


def export_image(collection, folder, meta_dict, scale=10, region=None):
    """ Batch export images to local directory, one image at a time
    Arguments
        :param collection: earth engine image collection object, a stack of images to export
        :param folder: specify the name of the folder to store images in
        :param scale=10: resolution of the image. Set to the orginal resolution of Sentinel 2 by default
        :param region=None: Area of interest to crop the image to.
                            If pass in coordinates, the image will be cropped to
                            interested range of coordinates
    """

    colList = collection.toList(collection.size())
    n = collection.size().getInfo()

    for i in range(n):
        img = ee.Image(colList.get(i))
        # imgid = img.id().getInfo()
        if region is None:
            region = img.geometry().bounds().getInfo()["coordinates"]

        batch.image.toLocal(
            image=img,
            name=folder,
            region=region,
            scale=scale
            )
        meta = img.getInfo()
        meta_dict[folder] = meta
def lc_code_to_str(code):
    """
    translates land cover codes to their names
    """
    if code == 11:
        return "open_water"
    elif 20 < code < 25:
        return "developed"
    elif code == 31:
        return "barren"
    elif 40 < code < 44:
        return "forest"
    elif code == 52:
        return "scrub"
    elif code == 71:
        return "grassland"
    elif code == 81:
        return "pasture"
    elif code == 82:
        return "crops"
    elif code == 90 or code == 95:
        return "wetlands"
    else:
        return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="path to csv input file", type=str)
    parser.add_argument(
        "-d",
        "--distance",
        help="side length for area of interest",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "-e",
        "--errorlog",
        help="path to error log file",
        default="data/error.log",
        type=str,
    )

    parser.add_argument(
        "-o", "--output_dir", help="path to output directory", default="data/", type=str
    )
    parser.add_argument(
        "-lat",
        "--lat_col",
        help="name of the column that contains the latitude",
        default="lat",
        type=str,
    )
    parser.add_argument(
        "-lon",
        "--lon_col",
        help="name of the column that contains the longitude",
        default="lon",
        type=str,
    )
    parser.add_argument(
        "-dc",
        "--domain_col",
        help="name of the column that contains the locations designated domain (land cover type by default)",
        default="LC_TYPE",
        type=str,
    )
    parser.add_argument(
        "-id",
        "--id_col",
        help="name of the column that contains the point id",
        default="rand_point_id",
        type=str,
    )

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        pass

    points = pd.read_csv(args.input)
    if "SAMPLE_LC1" in points.columns:
        points[args.domain_col] = points.SAMPLE_LC1.apply(lambda c: lc_code_to_str(c))
    points = points.dropna(subset=[args.domain_col])
    # with open('data/error.log', 'r') as f:
    #    lines = f.readlines()
    # error_points = set([int(l.split(':')[0].split()[-1]) for l in lines])

    logf = open(args.errorlog, "w")
    meta_dict = {}
    for lc in tqdm(points[args.domain_col].unique()):
        tmp = points[points[args.domain_col] == lc]

        for i, point in tqdm(tmp.iterrows()):
            fname = (
                f"{args.output_dir}/{point[args.domain_col]}_id_{point[args.id_col]}"
            )
            if os.path.exists(f"{fname}"):  # or point['rand_point_id'] in error_points:
                continue
            bbox = _bbox_from_point(
                (point[args.lat_col], point[args.lon_col]), args.distance
            )
            bbox = osm2gee_bbox(bbox)
            try:
                download_sen2_toLocal(bbox, fname, meta_dict)
                os.remove(f"{fname}.zip")
            except Exception as e:
                logf.write(f"point id {point[args.id_col]}: {e}\n")
                pass
            with open("meta", "wb") as f:
                pickle.dump(meta_dict, f)
                pass
