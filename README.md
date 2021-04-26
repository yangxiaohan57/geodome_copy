# GEODOME

GEODOME is a tool that the user to obtain open-source annotated Earth observation (EO) data and run image segmentation experiments with them using the mrs submodule.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all required pakages.

```bash
pip install -r requirements.txt
```

## Data Collection and Annotation

### Satellite Imagery Download

```bash
python
```

### OSM Labels Pipeline

```bash
python
```

### Rasterization Pipeline

```bash
python
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
