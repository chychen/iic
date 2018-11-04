"""
"""
import json
import tensorflow as tf
from PIL import Image
from data_augment import ImageNetPolicy
import numpy as np

NUM_CLASSES = 7178


# Create an image reader object for easy reading of the images
class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(
            self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


class Dataset(object):
    """ construct inputs for IIC(Inclusive Images Competition) training.
    """

    def __init__(self, train_data_path, validation_train_data_path,
                 validation_test_data_path, batch_size, buffer_size,
                 num_threads):
        LABEL_TO_CLASS_PATH = '../inputs/label_to_class.json'
        with open(LABEL_TO_CLASS_PATH, 'r') as infile:
            self.label_class_mapping = json.load(infile)
        self.dataset = {}
        ########################
        #### train ###
        self.dataset['train'] = tf.data.TFRecordDataset(train_data_path).map(
            self.decode_with_aug,
            num_threads).repeat().shuffle(buffer_size).batch(batch_size)
        ########################
        ### validation train ###
        self.dataset['validation_train'] = tf.data.TFRecordDataset(
            validation_train_data_path).map(
                self.decode_with_no_aug,
                num_threads).repeat().shuffle(buffer_size).batch(batch_size)
        ########################
        ### validation test ###
        self.dataset['validation_test'] = tf.data.TFRecordDataset(
            validation_test_data_path).map(
                self.decode_with_no_aug,
                num_threads).repeat().shuffle(1000).batch(batch_size)

        # create the iterators from the dataset
        self.train_iterator = self.dataset[
            'train'].make_initializable_iterator()
        self.vtrain_iterator = self.dataset[
            'validation_train'].make_initializable_iterator()
        self.vtest_iterator = self.dataset[
            'validation_test'].make_initializable_iterator()
        # same as in the doc https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.__iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.dataset['train'].output_types,
            self.dataset['train'].output_shapes)

    def decode_with_no_aug(self, serialized_example):
        return self.decode_with_aug(serialized_example, data_aug=False)

    def decode_with_aug(self, serialized_example, data_aug=True):
        # 3. Decode the record read by the reader
        feature = {
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64)
        }
        features = tf.parse_single_example(
            serialized_example, features=feature)
        # 4. Convert the image data from string back to the numbers
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        # labels are converted as bytes by array.array('H)
        label = tf.decode_raw(features['image/class/label'], tf.int16)
        label = tf.cast(label, tf.int32)
        # label is not sorted -> validate_indices=False
        label = tf.sparse_to_dense(
            label, (NUM_CLASSES, ), 1, 0, validate_indices=False)
        height = tf.cast(features['image/height'], tf.float32)
        width = tf.cast(features['image/width'], tf.float32)
        # 5. preprocessing
        ratio_w = width / 256
        ratio_h = height / 256
        ratio = tf.round(tf.maximum(ratio_w, ratio_h))
        image = tf.cast(image, tf.float32)
        image = tf.cond(ratio > 1, lambda: tf.image.resize_images(image, [tf.cast(height/ratio, tf.int32), tf.cast(width/ratio, tf.int32)]), lambda: image)
        if data_aug:
            image = tf.cast(image, tf.uint8)
            policy = ImageNetPolicy()
            image = tf.py_func(policy, [image], tf.uint8)
            image.set_shape(shape=(256, 256, 3))
            image = tf.cast(image, tf.float32)
            # image = tf.image.random_flip_left_right(image)
            # image = tf.image.random_brightness(image, max_delta=63)
            # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize_image_with_pad(image, 256, 256)
        return image, label

    def get_next(self):
        """
        Returns:
            Images: 4D tensor of [batch_size, height, width, 3].
            labels: 2D tensor of [batch_size, num_classes]
        """
        return self.__iterator.get_next()

    def get_label_to_human_readable(self):
        return self.label_class_mapping['label_to_human']

    def get_human_readable_to_label(self):
        return self.label_class_mapping['human_to_label']


# class TestDataset(object):
#     def __init__(self, test_data_path, num_threads=10, batch_size=64):
#         self.dataset = tf.data.TFRecordDataset(test_data_path).map(
#             self.decode, num_threads).batch(batch_size)
#         self.iterator = self.dataset.make_one_shot_iterator()

#     def get_next(self):
#         """
#         Returns:
#             Images: 4D tensor of [batch_size, height, width, 3].
#             labels: 2D tensor of [batch_size, num_classes]
#         """
#         return self.iterator.get_next()

#     def decode(self, serialized_example):
#         # 3. Decode the record read by the reader
#         feature = {
#             'image/encoded': tf.FixedLenFeature([], tf.string),
#             'image/class/label': tf.FixedLenFeature([], tf.string),
#             'image/height': tf.FixedLenFeature([], tf.int64),
#             'image/width': tf.FixedLenFeature([], tf.int64)
#         }
#         features = tf.parse_single_example(
#             serialized_example, features=feature)
#         # 4. Convert the image data from string back to the numbers
#         image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
#         # height = features['image/height']
#         # width = features['image/width']
#         # 5. preprocessing
#         image = tf.image.resize_image_with_pad(image, 256, 256)
#         image = tf.image.per_image_standardization(image)
#         return image

# Deprecated testing code
# def test():
#     LABEL_TO_CLASS_PATH = '../inputs/label_to_class.json'
#     with open(LABEL_TO_CLASS_PATH, 'r') as infile:
#         label_class_mapping = json.load(infile)
#     with tf.Session() as sess:
#         images, labels = get_next('../inputs/train_dataset.tfrecord', 32)
#         # Initialize all global and local variables
#         init_op = tf.group(tf.global_variables_initializer(),
#                            tf.local_variables_initializer())
#         sess.run(init_op)
#         # Create a coordinator and run all QueueRunner objects
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         print('threads', threads)
#         for _ in range(50):
#             img, lbl = sess.run([images, labels])
#             img = np.array(img)
#             print(img.shape)
#             for i in range(img.shape[0]):
#                 data = Image.fromarray(img[i], 'RGB')
#                 data.save('my_{}.png'.format(i))
#                 for i, v in enumerate(lbl[i]):
#                     if v != 0:
#                         print(label_class_mapping['label_to_human'][str(i)])
#                 input()

#         # Stop the threads
#         coord.request_stop()
#         # Wait for threads to stop
#         coord.join(threads)

if __name__ == '__main__':
    test()

# tf.TFRecordReader()
# def get_inputs(data_path, batch_size, mode='train', data_augmentation=True, buffer_size=100000, num_threads=16):
#     """ construct inputs for IIC(Inclusive Images Competition) training.

#     Args:
#         data_path: relative path to tfrecord files.
#         batch_size: number of images per batch.

#     Returns:
#         Images: 4D tensor of [batch_size, height, width, 3].
#         labels: 2D tensor of [batch_size, num_classes]
#     """
#     # 0. Create a list of filenames and pass it to a queue
#     filename_queue = tf.train.string_input_producer(
#         [data_path])
#     # 1&2. Define a reader and read the next record
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     # 3. Decode the record read by the reader
#     feature = {
#         'image/encoded': tf.FixedLenFeature([], tf.string),
#         'image/class/label': tf.FixedLenFeature([], tf.string),
#         'image/height': tf.FixedLenFeature([], tf.int64),
#         'image/width': tf.FixedLenFeature([], tf.int64)
#     }
#     features = tf.parse_single_example(
#         serialized_example, features=feature)
#     # 4. Convert the image data from string back to the numbers
#     image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
#     # labels are converted as bytes by array.array('H)
#     label = tf.decode_raw(features['image/class/label'], tf.int16)
#     label = tf.cast(label, tf.int32)
#     # label is not sorted -> validate_indices=False
#     label = tf.sparse_to_dense(
#         label, (NUM_CLASSES,), 1, 0, validate_indices=False)
#     # height = features['image/height']
#     # width = features['image/width']
#     # 5. preprocessing
#     image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
#     if data_augmentation:
#         image = tf.image.random_flip_left_right(image)
#         image = tf.image.random_brightness(image, max_delta=63)
#         image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
#         image = tf.image.per_image_standardization(image)
#     images, labels = tf.train.shuffle_batch(
#         [image, label], batch_size=batch_size, capacity=buffer_size+batch_size*num_threads, num_threads=num_threads, buffer_size=buffer_size)
#     return images, labels
