import math
from shapley_analyzer.prepare_input_data import code_features, find_all_feature_combination, calculate_margins_for_features, \
    noise_channels, decode_accuracy_map


def calculate_shapely(channel_name, channel_map):
    map_keys = channel_map.keys()
    number_of_channels = sum(len(x) == 1 for x in map_keys)
    channels_without_selected = [x for x in map_keys if all(y not in x for y in channel_name)]
    shapely_value = 0
    for i in channels_without_selected:
        char_list = [j for j in i]
        char_list.extend([j for j in channel_name])
        channels_with_selected = [x for x in map_keys if all(y in x for y in char_list) and len(x) == len(char_list)]
        if len(channels_with_selected) > 0:
            multiplicator = (math.factorial(len(i)) * math.factorial(number_of_channels - len(i) - 1) / math.factorial(
                number_of_channels))
            shapely_value += multiplicator * (channel_map[channels_with_selected[0]] - channel_map[i])
    return shapely_value


def calculate_shapely_for_model(feature_names, data, labels, model):
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
        noized_input = noise_channels(data, combination, all_features, feature_codes, margins)
        loss, accuracy = model.evaluate(noized_input, labels, batch_size=10)
        accuracy_map[combination] = accuracy
        # for nice representation
    accuracy_decoded_map = decode_accuracy_map(accuracy_map, feature_coding_map)

    # step 5: calculate shapely value for each channel
    shapely_map = {}
    for feature_code in feature_codes:
        shapely = calculate_shapely(feature_code, accuracy_map)
        shapely_map[feature_coding_map[feature_code]] = shapely
    return shapely_map, accuracy_decoded_map
