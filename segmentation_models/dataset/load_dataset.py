import inspect
import os

import numpy as np
import tensorflow as tf

PATH = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))


def load_dataset_for_training():
    path_train_original = os.path.join(PATH, 'data', 'train_original.npy')
    path_train_labeled = os.path.join(PATH, 'data', 'train_labeled.npy')

    train_original = np.load(path_train_original)
    train_labeled = np.load(path_train_labeled)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(train_original, tf.uint8) / 255, tf.cast(train_labeled, tf.uint8) / 255))

    path_test_original = os.path.join(PATH, 'data', 'test_original.npy')
    path_test_labeled = os.path.join(PATH, 'data', 'test_labeled.npy')

    test_original = np.load(path_test_original)
    test_labeled = np.load(path_test_labeled)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(test_original, tf.uint8) / 255, tf.cast(test_labeled, tf.uint8) / 255))
    return train_dataset, test_dataset


def load_test_files():
    path_test_original = os.path.join(PATH, 'data', 'test_original.npy')
    path_test_labeled = os.path.join(PATH, 'data', 'test_labeled.npy')
    test_original = np.load(path_test_original)
    test_labeled = np.load(path_test_labeled)
    return test_original, test_labeled


def load_10_validation_files():
    path_validated_original = os.path.join(PATH, 'data', 'validate_original_10.npy')
    path_validated_labeled = os.path.join(PATH, 'data', 'validate_labeled_10.npy')
    x_validate = np.load(path_validated_original)
    y_validate = np.load(path_validated_labeled)
    return x_validate, y_validate


def load_100_validation_files():
    path_validated_original = os.path.join(PATH, 'data', 'validate_original_100.npy')
    path_validated_labeled = os.path.join(PATH, 'data', 'validate_labeled_100.npy')
    print(path_validated_original)
    x_validate = np.load(path_validated_original)
    y_validate = np.load(path_validated_labeled)
    return x_validate, y_validate
