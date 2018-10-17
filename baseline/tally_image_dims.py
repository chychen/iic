""" According to 'tuning_labels.csv', she select labeled testing data from 'stage_1_test_images'.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import sys
import csv
import json
import array
import tensorflow as tf
from data_utils import ImageReader

# raw data path
TRAIN_IMAGE_FOLDER_PATH = '../inputs/train'
# contain validation and test
TEST_IMAGE_FOLDER_PATH = '../inputs/stage_1_test_images'
TRAIN_LABEL_CSV_PATH = '../labels/fixed_train_human_labels.csv'
# TRAIN_LABEL_CSV_PATH = '../labels/fixed_train_machine_labels.csv'
VALIDATION_LABEL_CSV_PATH = '../labels/tuning_labels.csv'

# to create `label_to_class.json`
CLASSES_CSV_PATH = '../labels/classes-trainable.csv'
# to create `label_to_class.json`
CLASSES_DES_CSV_PATH = '../labels/class-descriptions.csv'

# output
LABEL_TO_CLASS_PATH = '../inputs/label_to_class.json'
TFRECORD_TRAIN_PATH = '../inputs/train_dataset.tfrecord'
TFRECORD_VALIDATION_PATH = '../inputs/validation_dataset.tfrecord'
TFRECORD_TEST_PATH = '../inputs/test_dataset.tfrecord'


def tally_train_dataset():
    image_reader = ImageReader()
    # get validation filenames/labels by `tuning_labels.csv`
    filenames = os.listdir(TRAIN_IMAGE_FOLDER_PATH)
    max_h = -1
    max_w = -1
    batch_size = 100
    with tf.Session() as sess:
        for i in range(len(filenames)):
            image_data = tf.gfile.FastGFile(os.path.join(
                TRAIN_IMAGE_FOLDER_PATH, filenames[i]), 'rb').read()
            height, width = image_reader.read_image_dims(
                sess, image_data)
            if height > max_h:
                max_h = height
            if width > max_w:
                max_w = width
            if i % 1000 == 0:
                print('{}/{}'.format(i, len(filenames)))
                print(max_h, max_w)


def main():
    tally_train_dataset()


if __name__ == '__main__':
    main()
