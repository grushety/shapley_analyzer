# Explaining black box models through games
### -- first experiments using Shapley approach

The goal of this study project was to build and train a model that classifies flooded areas in remote sensing images.
and to estimate the contribution of each input feature (individual bands of the remote sensing images) to the classification result
through calculating the Shapley Value for each input feature.

We have tried to implement a simple library to calculate shapely values, noised the data and checked how valuable each feature's impact is.

### Models
**UNET**  
Unet represents one of the popular approaches for image segmentation models: to follow an encoder/decoder structure where we downsample the spatial resolution of the input, developing lower-resolution feature mappings which are learned to be highly efficient at discriminating between classes, and the upsample the feature representations into a full-resolution segmentation map.
![img/u-net-architecture.png](img/u-net-architecture.png)

**FCN**  
FCN is a network that does not contain any “Dense” layers (as in traditional CNNs) instead it contains 1x1 convolutions that perform the task of fully connected layers (Dense layers).
![img/fcn_architecture.png](img/fcn_architecture.png)

