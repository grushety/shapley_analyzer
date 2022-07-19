import matplotlib.pyplot as plt
import math


def show_images(image_arr, label_arr):
    n_lines = math.ceil(len(image_arr)/4)
    print(n_lines)
    fig, arr = plt.subplots(n_lines, 4, figsize=(15, 15))
    current_line = 0
    for i in range(len(image_arr)):
        print(current_line, i)
        arr[current_line, i-(current_line*4)].imshow(image_arr[i])
        arr[current_line, i-(current_line*4)].set_title(label_arr[i])
        if (1 + current_line)*4 <= i+1:
            current_line += 1
