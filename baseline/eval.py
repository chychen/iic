"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time
import json
import numpy as np
import pandas as pd
from tensorboard import summary as summary_lib
import tensorflow as tf
from absl import flags
import data_utils
import resnet_model

FLAGS = flags.FLAGS
flags.DEFINE_string('restore_path', None,
                    """path where to restore checkpoint.""")
flags.DEFINE_string('result_path', 'jay_result.csv',
                    """filepath where to save csv result.""")
flags.DEFINE_string(
    'test_data_folder_path', '../inputs/stage_1_test_images',
    """folder path of test images""")


def data_generator():
    test_image_filenames = os.listdir(FLAGS.test_data_folder_path)
    print(len(test_image_filenames))
    # Make a queue of file names including all the JPEG images files in the relative
    # image directory.
    test_image_paths = [os.path.join(
        FLAGS.test_data_folder_path, f_name) for f_name in test_image_filenames]
    filename_queue = tf.train.string_input_producer(
        test_image_paths, num_epochs=1, shuffle=False)
    # Read an entire image file which is required since they're JPEGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()
    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    image_name, image_file = image_reader.read(filename_queue)
    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image = tf.image.decode_jpeg(image_file, channels=3)
    # return image
    image = tf.image.resize_image_with_pad(image, 256, 256)
    image = tf.image.per_image_standardization(image)
    dict_ = tf.train.batch({'name': image_name, 'image': image}, 128, num_threads=20,
                           allow_smaller_final_batch=True)
    return dict_


def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.

    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.

    Args:
      resnet_size: The number of convolutional layers needed in the model.

    Returns:
      A list of block sizes to use in building the model.

    Raises:
      KeyError: if invalid resnet_size is received.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }
    return choices[resnet_size]


def eval():
    LABEL_TO_CLASS_PATH = '../inputs/label_to_class.json'
    with open(LABEL_TO_CLASS_PATH, 'r') as infile:
        label_class_mapping = json.load(infile)
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        data_dict = data_generator()
        batch_filenames = data_dict['name']
        batch_images = data_dict['image']
        # Calculate the gradients for each model tower.
        is_training = tf.placeholder(tf.bool)
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                model = resnet_model.Model(
                    resnet_size=18,
                    bottleneck=False,
                    num_classes=data_utils.NUM_CLASSES,
                    num_filters=64,
                    kernel_size=7,
                    conv_stride=2,
                    first_pool_size=3,
                    first_pool_stride=2,
                    block_sizes=_get_block_sizes(18),
                    block_strides=[1, 2, 2, 2],
                    resnet_version=resnet_model.DEFAULT_VERSION,
                    data_format='channels_last',
                    dtype=tf.float32)
                logits = model(batch_images, training=is_training)
                logits_round = tf.cast(tf.round(tf.sigmoid(logits)), tf.int32)
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            with tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run([init, init_local])
                # sess.run(iter_init)
                if FLAGS.restore_path is not None:
                    saver.restore(sess, FLAGS.restore_path)
                    print('successfully restore model from checkpoint: %s' %
                          (FLAGS.restore_path))
                # Create a coordinator and run all QueueRunner objects
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                predicted = []
                start_time = time.time()
                while True:
                    try:
                        filenames, predictions = sess.run([batch_filenames, logits_round],
                                                          feed_dict={is_training: False})
                        conds = np.not_equal(predictions, 0)
                        code_result = []
                        for cond in conds:
                            results = np.where(cond)
                            results = list(
                                map(lambda x: label_class_mapping['label_to_code'][str(x)], results[0]))
                            str_result = ''
                            for res in results:
                                str_result = str_result + ' ' + res
                            code_result.append(str_result)
                        filenames = list(
                            map(lambda x: os.path.split(x)[1][:-4], filenames))
                        for fname, result in zip(filenames, code_result):
                            predicted.append(
                                {'image_id': fname.decode('utf-8'), 'labels': result})
                        print(len(predicted))
                    except tf.errors.OutOfRangeError:
                        duration = time.time() - start_time
                        print('OutOfRangeError, and time cost: {}'.format(duration))
                        submission = pd.read_csv(
                            '../labels/stage_1_sample_submission.csv', index_col='image_id')
                        tuning_labels = pd.read_csv(
                            '../labels/tuning_labels.csv', names=['id', 'labels'], index_col=['id'])
                        predicted_df = pd.DataFrame.from_dict(
                            predicted, orient='columns')
                        predicted_df = predicted_df.set_index('image_id')
                        submission['labels'] = None
                        submission.update(predicted_df)
                        submission.update(tuning_labels)
                        submission.to_csv(FLAGS.result_path)
                        break
                # Stop the threads
                coord.request_stop()
                # Wait for threads to stop
                coord.join(threads)


def main(argv=None):
    eval()


if __name__ == '__main__':
    tf.app.run()
