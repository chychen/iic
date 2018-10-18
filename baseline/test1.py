import tensorflow as tf
import numpy as np


def main():
    with tf.Graph().as_default() as graph:
        # feedable iterator to switch between iterators
        EPOCHS = 10
        # making fake data using numpy
        train_data = (np.arange(100), np.arange(100))
        test_data = (np.arange(100), np.arange(100))
        # create placeholder
        x, y = tf.placeholder(
            tf.float32, shape=[
                None,
            ]), tf.placeholder(
                tf.float32, shape=[
                    None,
                ])
        # create two datasets, one for training and one for test
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        test_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        # create the iterators from the dataset
        train_iterator = train_dataset.make_initializable_iterator()
        test_iterator = test_dataset.make_initializable_iterator()
        # same as in the doc https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
        handle = tf.placeholder(tf.string, shape=[])
        iter = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
        next_elements = iter.get_next()
        with tf.Session() as sess:
            train_handle = sess.run(train_iterator.string_handle())
            test_handle = sess.run(test_iterator.string_handle())
            # initialise iterators.
            sess.run(
                train_iterator.initializer,
                feed_dict={
                    x: train_data[0],
                    y: train_data[1]
                })
            sess.run(
                test_iterator.initializer,
                feed_dict={
                    x: test_data[0],
                    y: test_data[1]
                })
            while True:
                for _ in range(EPOCHS):
                    x, y = sess.run(
                        next_elements, feed_dict={handle: train_handle})
                    print(x, y)
                x, y = sess.run(next_elements, feed_dict={handle: test_handle})
                print(x, y)
                input()


if __name__ == '__main__':
    main()
# import tensorflow as tf
# import numpy as np

# def main():
#     with tf.Graph().as_default() as graph:
#         # Reinitializable iterator to switch between Datasets
#         EPOCHS = 10
#         # making fake data using numpy
#         train_data = (np.arange(100), np.arange(100))
#         test_data = (np.arange(100), np.arange(100))
#         # create two datasets, one for training and one for test
#         train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
#         test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
#         # create a iterator of the correct shape and type
#         iter = tf.data.Iterator.from_structure(train_dataset.output_types,
#                                                train_dataset.output_shapes)
#         features, labels = iter.get_next()
#         # create the initialisation operations
#         train_init_op = iter.make_initializer(train_dataset)
#         test_init_op = iter.make_initializer(test_dataset)
#         with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#             while True:
#                 sess.run(train_init_op)  # switch to train dataset
#                 for _ in range(EPOCHS):
#                     print(sess.run([features, labels]))
#                 sess.run(test_init_op)  # switch to val dataset
#                 input(sess.run([features, labels]))

# if __name__ == '__main__':
#     main()

# import tensorflow as tf
# import numpy as np

# with tf.Session() as sess:
#     temp = tf.data.Dataset.range(100)
#     dataset = {}
#     dataset['a'] = temp.shard(10,0)
#     dataset['b'] = temp.shard(10,9)
#     dataset['a'] = dataset['a'].repeat().shuffle(5).batch(2)
#     dataset['b'] = dataset['b'].repeat().shuffle(5).batch(2)
#     iterator = {}
#     iterator['a'] = dataset['a'].make_one_shot_iterator()
#     iterator['b'] = dataset['b'].make_one_shot_iterator()
#     next_element = {}
#     next_element['a'] = iterator['a'].get_next()
#     next_element['b'] = iterator['b'].get_next()

#     while True:
#         value_a, value_b = sess.run([next_element['a'],next_element['b']])
#         print(value_a)
#         input(value_b)