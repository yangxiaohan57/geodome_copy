# GEODOME


## USAGE

### OSM PIPELINE

```
usage: osm_async_download.py [-h] [-i INPUT] [-o OUTFILE] [-ce CONTACT_EMAIL]
                             [-d DISTANCE] [-lat LAT_COL] [-lon LON_COL]
                             [-s SPLITS] [-t PAUSE_TIME]

optional arguments:
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
### GEE PIPELINE

```
usage: gee_tools.py [-h] [-i INPUT] [-d DISTANCE] [-e ERRORLOG]
                    [-o OUTPUT_DIR] [-lat LAT_COL] [-lon LON_COL]
                    [-dc DOMAIN_COL] [-id ID_COL]

optional arguments:
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

```
usage: preprocessing_pipeline.py [-h] [-i IN_DIR] [-o OUT_DIR] [-m META_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i IN_DIR, --in_dir IN_DIR
                        path to data directory
  -o OUT_DIR, --out_dir OUT_DIR
                        path to processed data directory
  -m META_FILE, --meta_file META_FILE
                        path to file containing meta information about the
                        data
```

### RASTERIZATION PIPELINE

```
usage: rasterize_labels.py [-h] [-i INPUT] [-ld LABEL_DICTIONARY]
                           [-lp LABEL_PRIORITY] [-m META] [-o OUT_DIR]
                           [-e ERRORLOG]

optional arguments:
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

