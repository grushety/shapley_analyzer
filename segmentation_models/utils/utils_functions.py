from random import randrange

import matplotlib.pyplot as plt
import math
import numpy as np


def show_images(image_arr, label_arr):
    n_lines = math.ceil(len(image_arr)/4)
    print(n_lines)
    fig, arr = plt.subplots(n_lines, 4, figsize=(15, 15))
    arr[-1, -1].axis('off')
    arr[-1, -2].axis('off')
    current_line = 0
    for i in range(len(image_arr)):
        arr[current_line, i-(current_line*4)].imshow(image_arr[i])
        arr[current_line, i-(current_line*4)].set_title(label_arr[i])
        if (1 + current_line)*4 <= i+1:
            current_line += 1


def noise_image(data, channels):
    images = np.copy(data)
    channels_to_noise = [i for i in "rgb" if(i not in channels)]

    # calculade range for channels based on real data
    r_min, r_max = np.min(images[:,:,:,0]), np.max(images[:,:,:,0])
    g_min, g_max = np.min(images[:,:,:,1]), np.max(images[:,:,:,1])
    b_min, b_max = np.min(images[:,:,:,2]), np.max(images[:,:,:,2])

    for image in images:
        for line in image:
            for pixel in line:
                if 'r' in channels_to_noise:
                    pixel[0] = randrange(r_min, r_max)
                if 'g' in channels_to_noise:
                    pixel[1] = randrange(g_min, g_max)
                if 'b' in channels_to_noise:
                    pixel[2] = randrange(b_min, b_max)
    return images