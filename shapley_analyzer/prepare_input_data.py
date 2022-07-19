import random
import string
import itertools

import numpy as np


def calculate_margins_for_features(feature_codes, input):
    margins = {}
    for feature_code in enumerate(feature_codes):
        margins[feature_code[1]] = {'min': input[feature_code[0]].min(), 'max': input[feature_code[0]].max()}
    return margins


# features are all channel to stay original
def noise_channels(input, features, all_features, feature_codes, margins, feature_type='none'):
    noised_input = np.copy(input)
    all = max(all_features, key=len)
    features_to_noise = [i for i in all if (i not in features)]
    for feature in enumerate(feature_codes):
        if feature[1] in features_to_noise:
            for item in noised_input:
                if feature_type != 'image':
                    item[feature[0]] = random.uniform(margins[feature[1]]['min'], margins[feature[1]]['max'])
                else:  # if a data is 3 dim image
                    for line in item:
                        for pixel in line:
                            pixel[feature[0]] = random.uniform(margins[feature[1]]['min'], margins[feature[1]]['max'])
    return noised_input


def code_features(feature_names):
    feature_coding_map = {}
    alphabet = list(string.ascii_lowercase)
    for feature_name in enumerate(feature_names):
        feature_coding_map[alphabet[feature_name[0]]] = feature_name[1]
    return feature_coding_map


def find_all_feature_combination(feature_codes):
    all_features = []
    for i in range(len(feature_codes)):
        combinations = set(itertools.combinations(feature_codes, i + 1))
        for feature_tuple in combinations:
            feature_combination = "".join(feature_tuple)
            all_features.append(feature_combination)
    return all_features


def decode_accuracy_map(accuracy_map, feature_coding_map):
    accuracy_decoded_map = {}
    for key in accuracy_map.keys():
        if (len(key) == 1):
            accuracy_decoded_map[feature_coding_map[key]] = accuracy_map[key]
        else:
            names = [feature_coding_map[i] for i in key]
            name = " & ".join(names)
            accuracy_decoded_map[name] = accuracy_map[key]
    return accuracy_decoded_map


def convert_map_to_arrays(map, feature_names, model_names):
    model_list = [[] for i in range(len(model_names))]
    for feature in feature_names:
        for model in enumerate(model_list):
            model_name = model_names[model[0]]
            model_info = map[model_name]
            value = model_info[feature]
            model[1].append(value)
    return model_list
