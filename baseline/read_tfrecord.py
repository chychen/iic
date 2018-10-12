import tensorflow as tf
from PIL import Image
import numpy as np


def main():
    with tf.Session() as sess:
        # 0. Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(
            ['../inputs/validation_dataset.tfrecord'], num_epochs=1)
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
        print(features['image/encoded'])
        # 4. Convert the image data from string back to the numbers
        # image = features['image/encoded']
        # image = tf.decode_raw(features['image/encoded'], tf.int8)
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        # image = tf.reshape(image, shape=[-1, -1, 3])
        print(image)
        # Cast label data into int32
        # label = tf.cast(features['image/class/label'], tf.int32)
        label = features['image/class/label']
        # Reshape image data into the original shape
        height = features['image/height']
        width = features['image/width']
        # image = tf.image.resize_image_with_crop_or_pad(image, 500, 500)
        print(height)
        print(label)
        print(image)
        # Any preprocessing here ...

        # Creates batches by randomly shuffling tensors
        # images, labels = tf.train.shuffle_batch(
        #     [image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(50):
            img, lbl, h, w = sess.run([image, label, height, width])
            img = np.array(img)
            img = np.reshape(img, [h, w, 3])
            data = Image.fromarray(img, 'RGB')
            data.save('my.png')
            print(img.shape)
            print(h)
            input(w)
            # Any preprocessing here ...

        # Stop the threads
        coord.request_stop()
        # Wait for threads to stop
        coord.join(threads)


if __name__ == '__main__':
    main()
