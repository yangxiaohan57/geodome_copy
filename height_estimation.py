from PIL import Image
from scipy import ndimage
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import math
import helper
import numpy as np

def connected_components(binary_image):
    """
    Execute Connected Components Function.
    """
    labeled, nr_objects = ndimage.label(binary_image)
    return labeled, nr_objects

def xy_coordinates(labeled):
    """
    Extract all the xy coordinates that has the same label as the stack base.
    """
    index = []
    for i in range(labeled.shape[0]):
        for j in range(labeled.shape[1]):
            if labeled[i,j] == labeled[labeled.shape[0] - 1, labeled.shape[1] - 1]:
                index.append([i,j])
    return index

def intersection_pt(intercept, slope, xy, labeled):
    """
    Find the intersection point with the smallest x and y values.
    """
    x_min = min([item[1] for item in xy])
    x_max = max([item[1] for item in xy])
    intersect = []
    for x in range(x_min, x_max +1):
        y = round(x*slope + intercept)
        if [y, x] in xy:
            intersect.append([y,x])
    return min(intersect)

def plot_cc(binary_image, labeled, slope, intersect, dir):
    """
    Plot Connected Component Graph
    """
    plt.figure(figsize=(10, 10))
    im = plt.imshow(labeled)

    # plot base of power plant
    plt.scatter(binary_image.shape[0], binary_image.shape[1], c='red', s=100, label='Stack Base')

    # plot intersection
    plt.scatter(intersect[1], intersect[0], c='black', s=100, label='Intersect')

    # angle lines
    plt.axline((binary_image.shape[0], binary_image.shape[1]), slope=slope, color='red',
               label='Est Shadow Trajectory')

    plt.axline((binary_image.shape[0] - 15, binary_image.shape[1]), slope=slope, color='red',
               label='Est Shadow Trajectory Error Bands', linestyle='dashed')

    plt.axline((binary_image.shape[0] + 15, binary_image.shape[1]), slope=slope, color='red',
               linestyle='dashed')

    values = np.unique(labeled.ravel())
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="Blob {l}".format(l=values[i])) for i in range(len(values))]
    # put those patched as legend-handles into the legend

    plt.legend(handles=patches, bbox_to_anchor=(0.5, 0., 1.1, 1), fontsize=18)
    # plt.legend(bbox_to_anchor=(0.5, 0., 1.1, 1), fontsize=18)
    plt.axis('off')
    plt.savefig(dir)

def dist(intersect, binary_image, slope):
    """
    Calculate the distance from the origin to the intersection
    """
    dist = math.sqrt( (binary_image.shape[1] - intersect[1])**2 + (binary_image.shape[0] - intersect[0])**2 )
    return dist

def calculate_stackheight(shadow_length, zn_angle, slope):
    """
    Calculate stack height
    """
    sl_ft = shadow_length * 3.28084
    elevation = 90 - zn_angle
    elevation_rad = math.radians(elevation)
    height = math.tan(sl_ft) * elevation_rad
    return height
