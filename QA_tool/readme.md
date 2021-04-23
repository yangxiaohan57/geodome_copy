### How to Use
#### if running from ssh on helios:
- open an ssh tunnel to helios with the following command:
  - `ssh -N -L 5500:localhost:5500 <user>@<hostname>`
- symlink the rasters and the images directories to the `static/` subfolder using the following command:
  - `ln -s full/source/dir/path full/path/geodome/QA_tool/static/<dirname>`
- be sure to have the complete path, not relative

#### if running locally:
- add the paths of the images directory and the rasters directory to the `img_dst` and `raster_dst` variables in `app.py`
- `cd QA_tool`
- `set FLASK_APP=app.py` (or `export FLASK_APP=app.py` depending on your OS)
- check that you set correctly by running `echo $FLASK_APP`
- `flask run --port=5500`

#### if you want to limit the checking to a specific set of images:
- add a csv file `fnames.csv` with 2 columns (fname, Decision) leave the `Decision` column empty and add the ids of the locations in the `fname` column