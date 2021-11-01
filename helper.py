# Contain the common functions used by all python files
def linear_graph(image_array, az_angle):
    """
    Function that calculates intercept and slope based on azimuth angle and image_array
    """
    shadow_angle = math.radians(90+abs(180-az_angle))
    intercept = (image_array.shape[1] - 1) -  (image_array.shape[0] - 1) * shadow_angle
    return intercept, shadow_angle

def plot(image_array, slope):
    """
    Function that draws the linear trend on the satellite imagery
    """
    fig, ax = plt.subplots(num='main', figsize=(10,10))
    plt.axis('off')
    plt.imshow(image_array, aspect = 'auto', cmap ='gray')
    plt.scatter(image_array.shape[0], image_array.shape[1], c='red', s=30, label = 'Stack Base')
    plt.axline((image_array.shape[0]-10, image_array.shape[1]), slope=slope, color='red',
               label='axline', linestyle='dashed')
    plt.axline((image_array.shape[0], image_array.shape[1]), slope=slope, color='red',
               label='axline')
    plt.axline((image_array.shape[0] + 10, image_array.shape[0]), slope=slope, color='red',
               label='axline', linestyle='dashed')