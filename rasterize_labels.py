"""A set of functions and command line tool to transform tags (key:values)
downloaded from OSM into the proposed hierarchical taxonomy of labels.
and rasterize them into an image that can be used as target for image segmentation task."""

import argparse
import json
import os
import pickle
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from skimage.io import imsave
from tqdm import tqdm

from osm_tools import DEFAULT_CRS, _create_gdf
from shapely.geometry import Polygon
import re

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# define constants
type_convert_dict = {'Key': str, 'Value': str}
startq = re.compile(r"(:\s|,\s|{)\'")
endq = re.compile(r"\'(,|:|})")

def get_clean_gdf(tags):
    """ turn the tags json into a gdf and dispose of rows without tags or geometry """
    tags_fixed_quotes = tags.replace("\\", "")
    tags_fixed_quotes = re.sub(r"\"", r"'", tags_fixed_quotes)
    tags_fixed_quotes = re.sub(startq, "\\1\"", tags_fixed_quotes)
    tags_fixed_quotes = re.sub(endq, "\"\\1", tags_fixed_quotes)
    tag_json = json.loads(tags_fixed_quotes)
    gdf = _create_gdf(tag_json, DEFAULT_CRS, False)
    try:
        cleaned_gdf = gdf.melt(
            id_vars=['osmid', 'geometry'], var_name='Key', value_name='Value')
    except KeyError:
        return None
    cleaned_gdf.dropna(how='any', inplace=True)
    cleaned_gdf.reset_index(drop=True, inplace=True)
    cleaned_gdf = cleaned_gdf.astype(type_convert_dict)
    return cleaned_gdf


def filter_tags(gdf, tags_to_keep):
    return pd.merge(gdf, tags_to_keep, on=["Key", "Value"])

def get_rasterized_labels(gdf, meta):
    img_size = (meta['height'], meta['width'])
    pixels = np.zeros(img_size, dtype=np.uint8)
    # make linestrings thicker than 1 pixel
    # 1 degree is approximately 111,111 meters, so we buffer for .00005 to get 11 meter wide lines
    gdf['geometry'] = gdf['geometry'].apply(lambda g: g.buffer(.00005) if g.geom_type == 'LineString' else g)
    for depth in sorted(gdf.Priority.unique())[::-1]:
        tmp = gdf[gdf.Priority == depth]
        raster = rasterize(shapes=tmp['geometry'],
                           out=pixels,
                           default_value=depth,
                           out_shape=img_size,
                           transform=(meta['transform']))
        pixels = raster
        pass
    return pixels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        help='csv with list of data collected from OSM.'
    )
    parser.add_argument(
        '-ld', '--label_dictionary',
        default='label_dictionary.csv',
        help='path to csv dictionary of tags to keep',
        type=str
    )

    parser.add_argument(
        '-lp', '--label_priority',
        default='label_priority.csv',
        help='path to csv dictionary of tag depths',
        type=str
    )

    parser.add_argument(
        '-m', '--meta',
        default='meta.pkl',
        help='path to pickle of meta dictionary',
        type=str
    )

    parser.add_argument('-o', '--out_dir',
                        help='path to rasterized tags directory',
                        default='./rasterized_tags',
                        type=str)

    parser.add_argument('-e', '--errorlog', 
                        help='path to error log file', 
                        default='./error.log', 
                        type=str)
    args = parser.parse_args()

    logf = open(args.errorlog, "w")

    LABEL_DICTIONARY = pd.read_csv(args.label_dictionary,
                                   names=['Key', 'Value', 'Main_Key', 'Main_Value',
                                          'Second_Key', 'Second_Value'],
                                   skiprows=1)

    LABEL_PRIORITY = pd.read_csv(args.label_priority,
                                 names=['Main_Key', 'Main_Value', 'Priority'], skiprows=1)

    tags_to_keep = pd.merge(LABEL_DICTIONARY,
                            LABEL_PRIORITY,
                            on=["Main_Key", "Main_Value"],
                            validate="m:1")

    input_df = pd.read_csv(args.input)
    
    with open(args.meta, 'rb') as f:
        meta_dict = pickle.load(f)
        pass

    if not os.path.exists(args.out_dir):
        print('making output directory')
        os.makedirs(args.out_dir)
        pass
    
    if not 'fname' in input_df.columns:
        input_df['fname'] = input_df['code'].astype(str) + '_id_' + input_df['rand_point_id'].astype(str)
        pass

    for _, location in tqdm(input_df.iterrows()):
        filename = "{}/{}_raster.png".format(args.out_dir, location['fname'])
        if os.path.exists(filename):
            continue
        meta = meta_dict.get(location['fname'], (0,0))[0]
        if meta == 0:
            logf.write("{} has no meta\n".format(location['fname']))
            continue
        try:
            gdf = get_clean_gdf(location['tags'])
            if((gdf is None) or (len(gdf) == 0)):
                logf.write("{} returned empty gdf\n".format(location['fname']))
                imsave(filename, np.zeros((meta['height'], meta['width']), dtype=np.int16))
                continue
            gdf = filter_tags(gdf, tags_to_keep)
            raster = get_rasterized_labels(gdf, meta)
            imsave(filename, raster)
        except Exception as e:
            logf.write("{} throw an error: {}\n\n".format(location['fname'], e))