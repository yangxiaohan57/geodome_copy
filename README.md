# GEODOME

GEODOME is a tool that the user to obtain open-source annotated Earth observation (EO) data and run image segmentation experiments with them using the mrs submodule. more information about the project is available in the [paper](documents/final_paper_GEODOME.pdf) and [presentation slides](documents/GEODOME_client_talk.pdf)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all required pakages.

```bash
pip install -r requirements.txt
```

## Data Collection and Annotation

### Satellite Imagery Download

to download the satellite imagery a google account that is registered to use the [Earth Engine API](https://developers.google.com/earth-engine) is needed. a typical usage example would be 

```bash
python gee_tools.py -i path/to/coordinates/file
```

This tool gets imagery from the NAIP satellite, to use other satellites, the `download_NAIP_toLocal` function would need to be replaced in the file with another satellite specific function.

additional arguments can be seen here:

```
usage: gee_tools.py [-h] [-i INPUT] [-d DISTANCE] [-e ERRORLOG]
                    [-o OUTPUT_DIR] [-lat LAT_COL] [-lon LON_COL]
                    [-dc DOMAIN_COL] [-id ID_COL]

arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to csv input file
  -d DISTANCE, --distance DISTANCE
                        side length for area of interest
  -e ERRORLOG, --errorlog ERRORLOG
                        path to error log file
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        path to output directory
  -lat LAT_COL, --lat_col LAT_COL
                        name of the column that contains the latitude
  -lon LON_COL, --lon_col LON_COL
                        name of the column that contains the longitude
  -dc DOMAIN_COL, --domain_col DOMAIN_COL
                        name of the column that contains the locations
                        designated domain (land cover type by default)
  -id ID_COL, --id_col ID_COL
                        name of the column that contains the point id
```

following is a simple pipeline that stacks the RGB bands and stores the tiff meta informations in a pickled dictionary.

```
usage: preprocessing_pipeline.py [-h] [-i IN_DIR] [-o OUT_DIR] [-m META_FILE]

arguments:
  -h, --help            show this help message and exit
  -i IN_DIR, --in_dir IN_DIR
                        path to data directory
  -o OUT_DIR, --out_dir OUT_DIR
                        path to processed data directory
  -m META_FILE, --meta_file META_FILE
                        path to file containing meta information about the
                        data
```

### OSM Labels Pipeline
This tool acquires OSM data and structure it as valid geoJSON.

```
usage: osm_async_download.py [-h] [-i INPUT] [-o OUTFILE] [-ce CONTACT_EMAIL]
                             [-d DISTANCE] [-lat LAT_COL] [-lon LON_COL]
                             [-s SPLITS] [-t PAUSE_TIME]

arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to csv input file
  -o OUTFILE, --outfile OUTFILE
                        path to csv output file
  -ce CONTACT_EMAIL, --contact_email CONTACT_EMAIL
                        contact email to include in the request header
  -d DISTANCE, --distance DISTANCE
                        side length for area of interest
  -lat LAT_COL, --lat_col LAT_COL
                        name of the column that contains the latitude
  -lon LON_COL, --lon_col LON_COL
                        name of the column that contains the longitude
  -s SPLITS, --splits SPLITS
                        number of splits to make to the input file
  -t PAUSE_TIME, --pause_time PAUSE_TIME
                        number of seconds to wait between request batches
```

### Rasterization Pipeline
This tool transforms OSM geoJSONS (output of OSM Label Pipeline) into bitmaps to be used as
the annotations of the objects of interest. The labels of interest can be easily
configured by editing a simple csv label mapper and passing it to the `-ld` and `-lp` arguments.

example use:

```
rasterize_labels.py -i path/to/osm_data -ld label_dictionary.csv -lp label_priority.csv -m path/to/meta -o /output/dir
```

additional arguments can be seen here:

```
usage: rasterize_labels.py [-h] [-i INPUT] [-ld LABEL_DICTIONARY]
                           [-lp LABEL_PRIORITY] [-m META] [-o OUT_DIR]
                           [-e ERRORLOG]

arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        csv with list of data collected from OSM.
  -ld LABEL_DICTIONARY, --label_dictionary LABEL_DICTIONARY
                        path to csv dictionary of tags to keep
  -lp LABEL_PRIORITY, --label_priority LABEL_PRIORITY
                        path to csv dictionary of tag depths
  -m META, --meta META  path to pickle of meta dictionary
  -o OUT_DIR, --out_dir OUT_DIR
                        path to rasterized tags directory
  -e ERRORLOG, --errorlog ERRORLOG
                        path to error log file
```

## Experiments

### Dataset Preprocessing

```bash
python model_scripts/preprocess.py
```

This tool takes as an input a directory with a set of RGB files, a directory with a set of OSM label rasters (ground truth images), and outputs a directory with preprocessed RGB files and label rasters in the appropriate format to be processed by the mrs models, a training file list and a validation file list as specified.

### Experiment Run Using mrs

```bash
python mrs/train.py
```

To train an mrs model with the generated GEODOME dataset, first specify a config.json file with the desired settings that indicates the directories of the data and the train and validation list files.
