""" According to 'tuning_labels.csv', she select labeled testing data from 'stage_1_test_images'.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import json
import tensorflow as tf
from data_utils import ImageReader

# raw data path
TRAIN_IMAGE_FOLDER_PATH = '../inputs/train'
# contain validation and test
TEST_IMAGE_FODLER_PATH = '../inputs/stage_1_test_images'
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




def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord():
    pass


def image_to_tfexample(image_data, image_format, height, width, class_id):
    feature = {
        'image/encoded': _bytes_feature(image_data),
        'image/format': _bytes_feature(image_format),
        'image/class/label': _int64_feature(class_id),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _convert_dataset(split_name, filenames, tfrecord_filepath):
    """Converts the given filenames to a TFRecord dataset.
    Args
    ----
        split_name: The name of the dataset, either 'train', 'validation' or 'test'.
        filenames: A list of absolute paths to png or jpg images.
    """
    assert split_name in ['train', 'validation', 'test']
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id, tfrecord_filename=tfrecord_filename, _NUM_SHARDS=_NUM_SHARDS)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i+1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(
                            filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(
                            sess, image_data)

                        class_name = os.path.basename(
                            os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())


def create_validation_test_dataset():
    with open(LABEL_TO_CLASS_PATH, 'r') as infile:
        label_class_mapping = json.load(infile)
    # get validation filenames/labels by `tuning_labels.csv`
    test_image_filenames = os.listdir(TEST_IMAGE_FODLER_PATH)
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
            validation_labels.append(labels)
    # correctness check
    for filename in validation_image_filenames:
        if filename not in test_image_filenames:
            raise FileNotFoundError(
                '{} is not found in test dataset'.format(filename))


def create_label_to_classes():
    """
    output
    ------
    label_to_class.json
        - label_to_class['label_to_human'] : label to human readable, such as label_to_class['label_to_human'][3110]='cloth'.
        - label_to_class['human_to_label'] : to label, such as label_to_class['human_to_label']['cloth']=3110.
        - label_to_class['label_to_code'] : label to code, such as label_to_class['label_to_code'][5991]='/m/fv809'.
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
    # create_train_dataset()
    create_validation_test_dataset()


if __name__ == '__main__':
    main()
