import ee
import numpy as np
import pandas as pd

import os

from osm_tools import _bbox_from_point, gdf_from_bbox
ee.Initialize()

from skimage.io import imread

from geetools import batch
from tqdm import tqdm

from argparse import ArgumentParser

def osm2gee_bbox(bbox):
    # convert bbox from lat lon to lon lat
    return bbox[1], bbox[0], bbox[3], bbox[2]


def download_NAIP_toLocal(bbox, name, scale=1):
    AOI = ee.Geometry.Rectangle(list(bbox), 
                                'EPSG:4326', 
                                False)

    collection = (ee.ImageCollection("USDA/NAIP/DOQQ")
                .filterDate('2010-01-01', '2019-01-01')
                .filterBounds(AOI)
                )

    image = ee.Image(collection.mosaic()).clip(AOI)
    batch.image.toLocal(image, name, scale=scale, region=AOI)



def lc_code_to_str(code):
    if code == 11:
        return 'open_water'
    elif 20<code <25:
        return 'developed'
    elif code == 31:
        return 'barren'
    elif 40 < code < 44:
        return 'forest'
    elif code == 52:
        return 'scrub'
    elif code == 71:
        return 'grassland'
    elif code == 81:
        return 'pasture'
    elif code == 82:
        return 'crops'
    elif code == 90 or code == 95:
        return 'wetlands'
    else:
        return None
    
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', 
                    help='path to csv input file', 
                    type=str)
    parser.add_argument('-d', '--distance', 
                        help='side length for area of interest', 
                        default=500, 
                        type=int)
    parser.add_argument('-e', '--errorlog', 
                        help='path to error log file', 
                        default='data/error.log', 
                        type=str)
    
    parser.add_argument('-o', '--output_dir', 
                        help='path to output directory', 
                        default='data/', 
                        type=str)
    parser.add_argument('-lat', '--lat_col', 
                        help='name of the column that contains the latitude', 
                        default='lat', 
                        type=str)
    parser.add_argument('-lon', '--lon_col', 
                        help='name of the column that contains the longitude', 
                        default='lon', 
                        type=str)
    parser.add_argument('-lc', '--land_cover_col', 
                        help='name of the column that contains the land cover type', 
                        default='LC_TYPE', 
                        type=str)
    parser.add_argument('-id', '--id_col', 
                        help='name of the column that contains the point id', 
                        default='rand_point_id', 
                        type=str)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        pass

    points = pd.read_csv(args.input)
    points[args.land_cover_col] = points.SAMPLE_LC1.apply(lambda c:lc_code_to_str(c))
    points = points.dropna(subset=[args.land_cover_col])
    # with open('data/error.log', 'r') as f:
    #    lines = f.readlines()
    # error_points = set([int(l.split(':')[0].split()[-1]) for l in lines])

    logf = open(args.errorlog, "w")

    for lc in tqdm(points.LC_TYPE.unique()):
        print(lc)
        tmp = points[points.LC_TYPE == lc]

        for i, point in tqdm(tmp.iterrows()):
            fname = f"{args.output_dir}/{point[args.land_cover_col]}_id_{point[args.id_col]}"
            if os.path.exists(f'{fname}'): # or point['rand_point_id'] in error_points:
                continue
            bbox = _bbox_from_point((point[args.lat_col], point[args.lon_col]), args.distance)
            bbox = osm2gee_bbox(bbox)
            try:
                download_NAIP_toLocal(bbox, fname)
                os.remove(f'{fname}.zip')
            except Exception as e:
                logf.write(f"point id {point[args.id_col]}: {e}\n")
                pass