import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    temp = tf.data.Dataset.range(100)
    dataset = {}
    dataset['a'] = temp.shard(10,0)
    dataset['b'] = temp.shard(10,9)
    dataset['a'] = dataset['a'].repeat().shuffle(5).batch(2)
    dataset['b'] = dataset['b'].repeat().shuffle(5).batch(2)
    iterator = {}
    iterator['a'] = dataset['a'].make_one_shot_iterator()
    iterator['b'] = dataset['b'].make_one_shot_iterator()
    next_element = {}
    next_element['a'] = iterator['a'].get_next()
    next_element['b'] = iterator['b'].get_next()

    while True:
        value_a, value_b = sess.run([next_element['a'],next_element['b']])
        print(value_a)
        input(value_b)