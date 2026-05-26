import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def load_image(image_path,show_info=False, plot_img=False):
    original = Image.open(image_path) 
    converted = original.convert("RGB") # standardize to RGB
    imgArr  = np.array(converted) # convert to numpy array
    if show_info:
        print(f"max and min pixel values: {imgArr.max()}, {imgArr.min()}")
        print(f"image shape: {imgArr.shape}")
    if plot_img:
        plt.imshow(imgArr)
        plt.show()
    return imgArr
