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
from tensorboard import summary as summary_lib
import tensorflow as tf
from absl import flags
import data_utils
import tf_utils
import resnet_model

FLAGS = flags.FLAGS
flags.DEFINE_string('comment', None,
                    """Why do you train? What do you want to verify?""")
flags.DEFINE_string('train_dir', None,
                    """Directory where to write event logs and checkpoint.""")
flags.DEFINE_string('restore_path', None,
                    """Directory where to restore checkpoint.""")
flags.DEFINE_string(
    'train_data_path', '../inputs/train_dataset_v2.tfrecord',
    """Path where store the training dataset, must be tfrecord format.""")
flags.DEFINE_string(
    'validation_train_data_path',
    '../inputs/validation_train_dataset_v2.tfrecord',
    """Path where store the training dataset, must be tfrecord format.""")
flags.DEFINE_string(
    'validation_data_path', '../inputs/validation_dataset.tfrecord',
    """Path where store the validation dataset, must be tfrecord format.""")
flags.DEFINE_float('lr', 1e-3, "learning rate.")
flags.DEFINE_float('MOVING_AVERAGE_DECAY', 0.9999,
                   """The decay to use for the moving average.""")
flags.DEFINE_integer('buffer_size', 100000,
                     """How many buffer size to make psuedo shuffle.""")
flags.DEFINE_integer('num_threads', 40, """How many threads for data Input.""")
flags.DEFINE_integer('num_gpus', 4, """How many GPUs to use.""")
flags.DEFINE_integer('batch_size', 256, """number of data per batch""")
flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")

LOG_COLLECTIONS = ['train', 'validation_train', 'validation_test']
# (Human)   Training Dataset: [Person:807k, Plant:267k, Street:0.4k, Art:2k]
# (Machine) Training Dataset: [Person:146k, Plant:114k, Street:55k, Art:114k]
# Stage One Tuning Dataset  : [Person:322, Plant:122, Street:63, Art:45]
TARGET_LABELS = ['Person', 'Plant', 'Street', 'Art']


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


def tower_loss(scope, images, labels, is_training, human_to_label):
    """Calculate the total loss on a single tower running the model.
    Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        images: Images. 4D tensor of shape [batch_size, height, width, 3].
        labels: Labels. 2D tensor of shape [batch_size, num_classes].
    Returns:
        Tensor of shape [] containing the total loss for a batch of data
    """
    # TODO scope? weight decay? validation_test? embedding? accuracy? saver? restored?
    # TODO Calculate loss, which includes softmax cross entropy and L2 regularization.

    # Build inference Graph.
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
    logits = model(images, training=is_training)
    # weights = tf.multiply(tf.cast(labels, tf.float32), 1000.0) + 1.0 # example: labels[0,0,1,1,0] -> weights[1,1,7179,7179,1]
    cross_entropy = tf.losses.sigmoid_cross_entropy(
        logits=logits,
        multi_class_labels=labels,
        weights=1.0,
        label_smoothing=0.0)

    # log summary of single gpu on tensorboard
    if 'tower_0' in scope:
        tf.summary.histogram('logits', logits, collections=LOG_COLLECTIONS)
        tf.summary.scalar(
            'loss_cross_entropy',
            cross_entropy,
            collections=LOG_COLLECTIONS,
            family='iic')

        logits_round = tf.cast(tf.round(tf.sigmoid(logits)), tf.int32)
        CM = tf_utils.Confusion_Matrix(logits_round, labels)
        tf.summary.scalar(
            'MEAN/f2_score',
            CM.f2_score,
            collections=LOG_COLLECTIONS,
            family='iic')
        tf.summary.scalar(
            'MEAN/precision',
            CM.precision,
            collections=LOG_COLLECTIONS,
            family='iic')
        tf.summary.scalar(
            'MEAN/recall',
            CM.recall,
            collections=LOG_COLLECTIONS,
            family='iic')

        for label_name in TARGET_LABELS:
            label_int = human_to_label[label_name]
            CM_target = tf_utils.Confusion_Matrix(logits_round[:, label_int],
                                                  labels[:, label_int])
            tf.summary.scalar(
                '{}/f2_score'.format(label_name),
                CM_target.f2_score,
                collections=LOG_COLLECTIONS,
                family=label_name)
            tf.summary.scalar(
                '{}/precision'.format(label_name),
                CM_target.precision,
                collections=LOG_COLLECTIONS,
                family=label_name)
            tf.summary.scalar(
                '{}/recall'.format(label_name),
                CM_target.recall,
                collections=LOG_COLLECTIONS,
                family=label_name)

    # # Assemble all of the losses for the current tower only.
    # losses = tf.get_collection('losses', scope)

    # # Calculate the total loss for the current tower.
    # total_loss = tf.add_n(losses, name='total_loss')

    # # Attach a scalar summary to all individual losses and the total loss; do the
    # # same for the averaged version of the losses.
    # for l in losses + [total_loss]:
    # # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # # session. This helps the clarity of presentation on tensorboard.
    # loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
    # tf.summary.scalar(loss_name, l)
    return cross_entropy


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.train.create_global_step(graph=graph)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        # batch_images, batch_labels = data_utils.get_inputs(
        #     FLAGS.train_data_path, FLAGS.batch_size//FLAGS.num_gpus)
        dataset = data_utils.Dataset(
            FLAGS.train_data_path,
            FLAGS.validation_train_data_path,
            FLAGS.validation_data_path,
            FLAGS.batch_size // FLAGS.num_gpus,
            buffer_size=FLAGS.buffer_size,
            num_threads=FLAGS.num_threads)
        batch_images, batch_labels = dataset.get_next()
        tf.summary.image(
            'train images', batch_images, collections=LOG_COLLECTIONS)
        # Calculate the gradients for each model tower.
        tower_grads = []
        is_training = tf.placeholder(tf.bool)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % (i)) as scope:
                        loss = tower_loss(
                            scope, batch_images, batch_labels, is_training,
                            dataset.get_human_readable_to_label())
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = optimizer.compute_gradients(loss)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
        # summary
        summaries = tf.get_collection('train')
        vtrain_summaries = tf.get_collection('validation_train')
        vtest_summaries = tf.get_collection('validation_test')
        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', FLAGS.lr))
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram(var.op.name + '/gradients', grad))
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(
            grads, global_step=global_step)
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        # Group all updates to into a single train op.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op, variables_averages_op)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        summary_op = tf.summary.merge(summaries)
        vtrain_summary_op = tf.summary.merge(vtrain_summaries)
        vtest_summary_op = tf.summary.merge(vtest_summaries)
        init = tf.global_variables_initializer()
        num_batch_per_epoch = int(1.7e6 // FLAGS.batch_size)
        with tf.Session(
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=FLAGS.log_device_placement)) as sess:
            sess.run(init)
            if FLAGS.restore_path is not None:
                saver.restore(sess, FLAGS.restore_path)
                print('successfully restore model from checkpoint: %s' %
                      (FLAGS.restore_path))
            train_handle, vtrain_handle, vtest_handle = sess.run([
                dataset.train_iterator.string_handle(),
                dataset.vtrain_iterator.string_handle(),
                dataset.vtest_iterator.string_handle()
            ])
            sess.run([
                dataset.train_iterator.initializer,
                dataset.vtrain_iterator.initializer,
                dataset.vtest_iterator.initializer
            ])
            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.train_dir, 'train'), sess.graph)
            vtrain_summary_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.train_dir, 'validation_train'))
            vtest_summary_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.train_dir, 'validation_test'))
            batch_idx = 0
            while True:
                start_time = time.time()
                _, loss_value, global_step_v, summary_str = sess.run(
                    [train_op, loss, global_step, summary_op],
                    feed_dict={
                        dataset.handle: train_handle,
                        is_training: True
                    })
                batch_idx = global_step_v // FLAGS.num_gpus
                duration = time.time() - start_time
                if global_step_v % (
                        100 * FLAGS.num_gpus  # per 100 batches
                ) == 0 or global_step_v == 0:
                    vtrain_loss_value, vtrain_summary_str = sess.run(
                        [loss, vtrain_summary_op],
                        feed_dict={
                            dataset.handle: vtrain_handle,
                            is_training: False
                        })
                    vtest_loss_value, vtest_summary_str = sess.run(
                        [loss, vtest_summary_op],
                        feed_dict={
                            dataset.handle: vtest_handle,
                            is_training: False
                        })
                    examples_per_sec = FLAGS.batch_size / duration
                    sec_per_batch = duration
                    format_str = (
                        '%s: batch_id %d, epoch_id %d, loss = %f vtrain_loss_value = %f vtest_loss_value = %f (%.2f examples/sec; %.2f sec/batch)'
                    )
                    print(format_str %
                          (datetime.now(), batch_idx, batch_idx //
                           num_batch_per_epoch, loss_value, vtrain_loss_value,
                           vtest_loss_value, examples_per_sec, sec_per_batch))
                    summary_writer.add_summary(summary_str, batch_idx)
                    vtrain_summary_writer.add_summary(vtrain_summary_str,
                                                      batch_idx)
                    vtest_summary_writer.add_summary(vtest_summary_str,
                                                     batch_idx)
                # Save the model checkpoint periodically.
                if global_step_v % (2 * num_batch_per_epoch *
                                    FLAGS.num_gpus) == 0:  # per 2 epochs
                    checkpoint_path = os.path.join(FLAGS.train_dir,
                                                   'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=batch_idx)
                    print('successfully save model!')
            # Stop the threads
            coord.request_stop()
            # Wait for threads to stop
            coord.join(threads)


def main(argv=None):
    assert FLAGS.comment is not None, "Please comments, for example why do you train? what do you want to verify?"
    if FLAGS.restore_path is None:
        if tf.gfile.Exists(FLAGS.train_dir):
            ans = input('"%s" will be removed!! are you sure (y/N)? ' %
                        FLAGS.train_dir)
            if ans == 'Y' or ans == 'y':
                # when not restore, remove follows (old) for new training
                tf.gfile.DeleteRecursively(FLAGS.train_dir)
                print('rm -rf "%s" complete!' % FLAGS.train_dir)
            else:
                exit()
        tf.gfile.MakeDirs(FLAGS.train_dir)
        with open(os.path.join(FLAGS.train_dir, 'config.json'), 'w') as out:
            json.dump(flags.FLAGS.flag_values_dict(), out)
        print(flags.FLAGS.flag_values_dict().items())
    train()


if __name__ == '__main__':
    tf.app.run()
