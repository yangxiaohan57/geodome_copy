import numpy as np
import matplotlib.pyplot as plt

def threshold(image_array, thresh):
    """
    Convert to binary based on the input threshold
    """
    return(np.where(image_array < thresh, 0, 1))

def plot_threshold(image_array, slope, dir):
    """
    Function that draws the linear trend on the satellite imagery
    """
    fig, ax = plt.subplots(num='main', figsize=(10,10))
    plt.axis('off')
    plt.imshow(image_array, aspect = 'auto', cmap ='gray')
    plt.scatter(image_array.shape[0], image_array.shape[1], c='red', s=30, label = 'Stack Base')
    plt.axline((image_array.shape[0]-10, image_array.shape[1]), slope=slope, color='red',
                linestyle='dashed')
    plt.axline((image_array.shape[0], image_array.shape[1]), slope=slope, color='red',
                label='Shadow Trajectory')
    plt.axline((image_array.shape[0] + 10, image_array.shape[0]), slope=slope, color='red',
                linestyle='dashed')
    plt.legend(fontsize=18)
    plt.savefig(dir)
    plt.close()
