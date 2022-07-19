# Shapley Analyzer
## _Package to calculate Shapley Value of NN_

This package calculates shape value for a neural network model


To calculate Shapely value for model we can use the **calculate_shapely_for_model** function with the following arguments:
1) **features_names** - an array of strings containing the names of all features in the correct order
2) **test_data_x** - test input to the model
3) **test_data_y** - test data labels
4) **model** - trained and compiled tensorflow model
5) **feature_type**: one option, if image then feature_type = "image", otherwise "none".