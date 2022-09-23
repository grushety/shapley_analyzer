import math
import tensorflow as tf
from prepare_input_data import code_features, find_all_feature_combination, \
    calculate_margins_for_features, \
    noise_channels, decode_accuracy_map


def calculate_shapley(channel_name, channel_map):
    map_keys = channel_map.keys()
    number_of_channels = sum(len(x) == 1 for x in map_keys)
    channels_without_selected = [x for x in map_keys if all(y not in x for y in channel_name)]
    shapley_value = 0
    for i in channels_without_selected:
        char_list = [j for j in i]
        char_list.extend([j for j in channel_name])
        channels_with_selected = [x for x in map_keys if all(y in x for y in char_list) and len(x) == len(char_list)]
        if len(channels_with_selected) > 0:
            multiplicator = (math.factorial(len(i)) * math.factorial(number_of_channels - len(i) - 1) / math.factorial(
                number_of_channels))
            shapley_value += multiplicator * (channel_map[channels_with_selected[0]] - channel_map[i])
    return shapley_value


def calculate_shapley_for_model(feature_names, data, labels, model, feature_type='none'):
    if feature_type == 'image':
        labels = tf.cast(labels / 255, tf.float32)

    # step 1: encode features/channels with alphabet letters
    feature_coding_map = code_features(feature_names)
    feature_codes = feature_coding_map.keys()

    # step 2: calculate all possible combination of channels
    all_features = find_all_feature_combination(feature_codes)

    # step 3: calculate min and max for each feature for given set
    margins = calculate_margins_for_features(feature_codes, data)

    # step 4: evaluate model for each clean combination of channels
    accuracy_map = {}
    for combination in all_features:
        noised_input = noise_channels(data, combination, all_features, feature_codes, margins,
                                      feature_type=feature_type)
        if feature_type == 'image':
            noised_input = noised_input / 255
            noised_input = tf.cast(noised_input, tf.float32)
        loss, accuracy = model.evaluate(noised_input, labels, batch_size=10)
        accuracy_map[combination] = accuracy

    # for nice representation
    accuracy_decoded_map = decode_accuracy_map(accuracy_map, feature_coding_map)

    # step 5: calculate shapley value for each channel
    shapley_map = {}
    for feature_code in feature_codes:
        shapley = calculate_shapley(feature_code, accuracy_map)
        shapley_map[feature_coding_map[feature_code]] = shapley
    return shapley_map, accuracy_decoded_map
