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
from random import shuffle
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
TFRECORD_TRAIN_PATH = '../inputs/train_dataset_v2.tfrecord'
TFRECORD_VALIDATION_TRAIN_PATH = '../inputs/validation_train_dataset_v2.tfrecord'
TFRECORD_VALIDATION_PATH = '../inputs/validation_dataset.tfrecord'
TFRECORD_TEST_PATH = '../inputs/test_dataset.tfrecord'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_to_tfexample(image_data, height, width, class_ids):
    feature = {
        'image/encoded': _bytes_feature(image_data),
        'image/class/label': _bytes_feature(class_ids),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def convert_to_tfrecord(split_name, filenames, labels):
    """Converts the given filenames to a TFRecord dataset.
    Args
    ----
        split_name : The name of the dataset, either 'train', 'validation' or 'test'.
        filenames :  A list of image filesnames.
        labels : A list of labels of the images.
    """
    assert split_name in ['train', 'validation_train', 'validation', 'test']
    if split_name == 'test':
        assert labels is None
    outfile_path = {
        'train': TFRECORD_TRAIN_PATH,
        'validation_train': TFRECORD_VALIDATION_TRAIN_PATH,
        'validation': TFRECORD_VALIDATION_PATH,
        'test': TFRECORD_TEST_PATH
    }
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            with tf.python_io.TFRecordWriter(outfile_path[split_name]) as tfrecord_writer:
                print('Preparing {} dataset'.format(split_name))
                num_slot = 20
                for i in range(len(filenames)):
                    # Progress Bar
                    cursor = int(i/len(filenames) * num_slot) + 1
                    sys.stdout.write('\r')
                    sys.stdout.write(
                        ">> Converting image [%-20s] %d/%d %.2f%%" % ('='*cursor, i+1, len(filenames), int(100.0*(i+1)/len(filenames))))
                    sys.stdout.flush()

                    # Read the filename:
                    if split_name == 'test':
                        class_ids = b''
                        image_root_path = TEST_IMAGE_FOLDER_PATH
                    elif split_name == 'validation':
                        class_ids = labels[i]  # multi labels
                        image_root_path = TEST_IMAGE_FOLDER_PATH
                    else:
                        class_ids = labels[i]  # multi labels
                        image_root_path = TRAIN_IMAGE_FOLDER_PATH
                    image_data = tf.gfile.FastGFile(os.path.join(
                        image_root_path, filenames[i]), 'rb').read()
                    height, width = image_reader.read_image_dims(
                        sess, image_data)
                    example = image_to_tfexample(
                        image_data, height, width, class_ids)

                    # NOTE: get very wierd result-> store decoded data into TFRecord getting smaller file size..., however I store raw data instead

                    tfrecord_writer.write(example.SerializeToString())
                sys.stdout.write('\n')
                sys.stdout.flush()


def create_validation_test_dataset():
    with open(LABEL_TO_CLASS_PATH, 'r') as infile:
        label_class_mapping = json.load(infile)
    # get validation filenames/labels by `tuning_labels.csv`
    test_image_filenames = os.listdir(TEST_IMAGE_FOLDER_PATH)
    validation_image_filenames = []
    validation_labels = []
    file_reader = csv.reader(
        open(VALIDATION_LABEL_CSV_PATH, 'r'), delimiter=',')
    for i, row in enumerate(file_reader):
        if i != 0:  # first row is ['ImageID', 'LabelNamesss']
            label_codes = str(row[1]).split(' ')
            validation_image_filenames.append(row[0]+'.jpg')
            labels = [label_class_mapping['code_to_label'][code]
                      for code in label_codes]
            # H: unsigned short, encode to byte code
            validation_labels.append(array.array('H', labels).tostring())
    # counterrr = 0
    for filename in validation_image_filenames:
        if filename not in test_image_filenames:
            raise FileNotFoundError(
                '{} is not found in test dataset'.format(filename))
    # convert to tfrecord
    convert_to_tfrecord(
        'validation', validation_image_filenames, validation_labels)
    convert_to_tfrecord('test', test_image_filenames, None)


def create_train_dataset():
    with open(LABEL_TO_CLASS_PATH, 'r') as infile:
        label_class_mapping = json.load(infile)
    # get validation filenames/labels by `tuning_labels.csv`
    train_image_filenames = os.listdir(TRAIN_IMAGE_FOLDER_PATH)
    shuffle(train_image_filenames)
    train_labels_dict = {}
    file_reader = csv.reader(
        open(TRAIN_LABEL_CSV_PATH, 'r'), delimiter=',')
    for i, row in enumerate(file_reader):
        if i != 0:  # first row is ['ImageID', 'Source', 'LabelName', 'Confidence']
            label_codes = row[2]
            filename = row[0]+'.jpg'
            if filename not in train_labels_dict.keys():
                train_labels_dict[filename] = [
                    label_class_mapping['code_to_label'][label_codes]]
            else:
                train_labels_dict[filename].append(
                    label_class_mapping['code_to_label'][label_codes])
    ordered_train_labels = []
    temp = copy.deepcopy(train_image_filenames)
    not_exist_counter = 0
    for filename in temp:
        if filename not in train_labels_dict.keys():
            not_exist_counter += 1
            train_image_filenames.remove(filename)
        else:
            # H: unsigned short, encode to byte code
            ordered_train_labels.append(array.array(
                'H', train_labels_dict[filename]).tostring())
    print('There are {} training images not exist in label csv file'.format(
        not_exist_counter))
    # convert to tfrecord
    train_data_amount = len(train_image_filenames)
    print('training dataset amount::', train_data_amount)
    convert_to_tfrecord(
        'train', train_image_filenames[:train_data_amount*9//10], ordered_train_labels[:train_data_amount*9//10])
    convert_to_tfrecord(
        'validation_train', train_image_filenames[train_data_amount*9//10:], ordered_train_labels[train_data_amount*9//10:])


def create_label_to_classes():
    """
    output
    ------
    label_to_class.json
        - label_to_class['label_to_human'] : label to human readable, such as label_to_class['label_to_human']["3110"]='cloth'.
        - label_to_class['human_to_label'] : to label, such as label_to_class['human_to_label']['cloth']=3110.
        - label_to_class['label_to_code'] : label to code, such as label_to_class['label_to_code']["5991"]='/m/fv809'.
        - label_to_class['code_to_label'] : to label, such as label_to_class['code_to_label']['/m/fv809']=5991.
    """
    description_file_reader = csv.reader(
        open(CLASSES_DES_CSV_PATH, 'r', encoding='utf-8'), delimiter=',')
    code_to_class_mapping = {}
    for i, row in enumerate(description_file_reader):
        if i != 0:  # first row is ['label_code', 'description']
            code_to_class_mapping[row[0]] = row[1]

    file_reader = csv.reader(open(CLASSES_CSV_PATH, 'r'), delimiter=',')
    trainable_classes_list = []
    for i, row in enumerate(file_reader):
        if i != 0:  # first row is ['labelcode']
            trainable_classes_list.append(row[0])
    label_to_class_mapping = {}
    label_to_class_mapping['label_to_code'] = dict(
        zip(range(len(trainable_classes_list)), trainable_classes_list))
    label_to_class_mapping['code_to_label'] = dict(
        zip(trainable_classes_list, range(len(trainable_classes_list))))
    label_to_class_mapping['label_to_human'] = dict(
        zip(range(len(trainable_classes_list)), [code_to_class_mapping[c] for c in trainable_classes_list]))
    label_to_class_mapping['human_to_label'] = dict(
        zip([code_to_class_mapping[c] for c in trainable_classes_list], range(len(trainable_classes_list))))

    with open(LABEL_TO_CLASS_PATH, 'w') as outfile:
        json.dump(label_to_class_mapping, outfile)
    # print('{}'.format(label_to_class_mapping).encode('utf-8'))


def main():
    # create_label_to_classes()
    # create_validation_test_dataset()
    create_train_dataset()


if __name__ == '__main__':
    main()
