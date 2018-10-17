import json
import numpy as np
import tensorflow as tf
from PIL import Image


def main():
    LABEL_TO_CLASS_PATH = '../inputs/label_to_class.json'
    with open(LABEL_TO_CLASS_PATH, 'r') as infile:
        label_class_mapping = json.load(infile)
    with tf.Session() as sess:
        # 0. Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(
            ['../inputs/train_dataset.tfrecord'], num_epochs=1)
        # 1&2. Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
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
        # Cast label data into int32
        label = tf.cast(label, tf.int32)
        num_labels = len(label_class_mapping['label_to_human'].keys())
        # label is not sorted -> validate_indices=False
        label = tf.sparse_to_dense(label, (num_labels,), 1, 0, validate_indices=False)
        print(label)
        # Reshape image data into the original shape
        height = features['image/height']
        width = features['image/width']
        image = tf.image.resize_image_with_crop_or_pad(image, 1024, 1024)
        # Any preprocessing here ...

        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=32, capacity=500, num_threads=4, min_after_dequeue=100)
        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(50):
            img, lbl = sess.run([images, labels])
            img = np.array(img)
            print(img.shape)
            for i in range(img.shape[0]):
                data = Image.fromarray(img[i], 'RGB')
                data.save('my_{}.png'.format(i))
                for i, v in enumerate(lbl[i]):
                    if v != 0:
                        print(label_class_mapping['label_to_human'][str(i)])
                input()
            # Any preprocessing here ...

        # Stop the threads
        coord.request_stop()
        # Wait for threads to stop
        coord.join(threads)


if __name__ == '__main__':
    main()
