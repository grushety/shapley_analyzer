# Dataset

In this project, we consider the **segmentation problem**,   
which consists in separating flooded areas from non-flooded areas.   
The data set consists of monitoring data from satellites,   
which are presented as an RGB image. 
Label set is mask with flooded area written as 255 and non-flooded area as 0.

Training origin and labeled data are converted to a numpy array and saved as .npy files.
The original image corresponds to the labeled image with the same array index in the corresponding file.

The *training dataset* contains **5000** items, the *test dataset* contains **1000** items,
and two *validation datasets* have 100 and 10 items.

Data are available to download as standard RGB PNG images  
Train data (x, input):  
https://nextcloud.gfz-potsdam.de/s/eedFKr774BNGZDj/download  
Label data (y, target):  
https://nextcloud.gfz-potsdam.de/s/3mjcgyJEqWCqWQq/download  
Validation data (x, input):  
https://nextcloud.gfz-potsdam.de/s/skqHjqKCAixKQsT/download  
Validation labels (y, target):  
https://nextcloud.gfz-potsdam.de/s/m6qXR3kN4yPmDRf/download  