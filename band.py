from imgProcessing import *

# Load Metadata
with open("data/meta", "rb") as f:
    meta_data = pickle.load(f)

# Get all get_plant_names
plant_names = meta_data.keys()
data = []
for pplant_name in plant_names:
    # Loop through the images in each plant
    img_names = get_all_img_names(meta_data, pplant_name)
    perform_band_stacking(img_names, pplant_name)
