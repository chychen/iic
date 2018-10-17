""" 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf
from absl import flags
import data_utils
import resnet_model

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', './baseline_bn',
                    """Directory where to write event logs and checkpoint.""")
flags.DEFINE_string('train_data_path', '../inputs/train_dataset.tfrecord',
                    """Path where store the training dataset, must be tfrecord format.""")
flags.DEFINE_string('validation_data_path', '../inputs/validation_dataset.tfrecord',
                    """Path where store the validation dataset, must be tfrecord format.""")
flags.DEFINE_float('lr', 1e-3,
                   "learning rate.")
flags.DEFINE_float('MOVING_AVERAGE_DECAY', 0.9999,
                   """The decay to use for the moving average.""")
flags.DEFINE_integer('num_gpus', 1,
                     """How many GPUs to use.""")
flags.DEFINE_integer('batch_size', 128,
                     """number of data per batch""")
flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")


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


def tower_loss(scope, logits, labels, mode, resnet_size=50):
    """Calculate the total loss on a single tower running the model.
    Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        labels: Labels. 2D tensor of shape [batch_size, num_classes].
    Returns:
        Tensor of shape [] containing the total loss for a batch of data
    """
    # TODO scope? weight decay? validation_test? embedding? accuracy? saver? restored?

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.sigmoid_cross_entropy(
        logits=logits, multi_class_labels=labels)
    tf.summary.scalar('loss_cross_entropy', cross_entropy, collections=[mode])

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
            FLAGS.train_data_path, FLAGS.validation_data_path, FLAGS.batch_size//FLAGS.num_gpus)
        batch_images, batch_labels = dataset.get_next('train')
        tf.summary.image(
            'train images', batch_images, collections=['train'])
        vtrain_batch_images, vtrain_batch_labels = dataset.get_next(
            'validation_train')
        tf.summary.image(
            'validation train images', vtrain_batch_images, collections=['validation_train'])
        vtest_batch_images, vtest_batch_labels = dataset.get_next(
            'validation_test')
        tf.summary.image(
            'validation test images', vtest_batch_images, collections=['validation_test'])
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % (i)) as scope:
                        # Build inference Graph.
                        model = resnet_model.Model(resnet_size=18,
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
                        ######################
                        ##       train      ##
                        logits = model(batch_images, training=True)
                        loss = tower_loss(scope, logits, batch_labels, mode='train')
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        ######################
                        ## validation_train ##
                        vtrain_logits = model(vtrain_batch_images, training=True)
                        vtrain_loss = tower_loss(
                            scope, vtrain_logits, vtrain_batch_labels, mode='validation_train')
                        ######################
                        ##  validation_test ##
                        vtest_logits = model(vtest_batch_images, training=False)
                        vtest_loss = tower_loss(
                            scope, vtest_logits, vtest_batch_labels, mode='validation_test')
                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = optimizer.compute_gradients(loss)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
        # summary
        summaries = tf.get_collection(
            'train')
        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', FLAGS.lr))
        vtrain_summaries = tf.get_collection(
            'validation_train')
        vtest_summaries = tf.get_collection(
            'validation_test')
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(
                    var.op.name + '/gradients', grad))

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
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge(summaries)
        vtrain_summary_op = tf.summary.merge(vtrain_summaries)
        vtest_summary_op = tf.summary.merge(vtest_summaries)
        init = tf.global_variables_initializer()
        num_batch_per_epoch = int(1.7e6//(FLAGS.batch_size*FLAGS.num_gpus))
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement)) as sess:
            sess.run(init)
            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.train_dir, 'train'), sess.graph)
            vtrain_summary_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.train_dir, 'validation_train'), sess.graph)
            vtest_summary_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.train_dir, 'validation_test'), sess.graph)
            batch_idx = 0
            while True:
                start_time = time.time()
                _, loss_value, global_step_v = sess.run(
                    [train_op, loss, global_step])
                batch_idx = global_step_v // FLAGS.num_gpus
                duration = time.time() - start_time
                # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if global_step_v % (100*FLAGS.num_gpus) == 0:
                    vtrain_loss_value, vtest_loss_value = sess.run(
                        [vtrain_loss, vtest_loss])
                    examples_per_sec = FLAGS.batch_size / duration
                    sec_per_batch = duration
                    format_str = (
                        '%s: batch_id %d, epoch_id %d, loss = %f vtrain_loss_value = %f vtest_loss_value = %f (%.2f examples/sec; %.2f sec/batch)')
                    print(format_str % (datetime.now(), batch_idx, batch_idx // num_batch_per_epoch,
                                        loss_value, vtrain_loss_value, vtest_loss_value, examples_per_sec, sec_per_batch))
                if global_step_v % (100*FLAGS.num_gpus) == 0:
                    summary_str, vtrain_summary_str, vtest_summary_str = sess.run(
                        [summary_op, vtrain_summary_op, vtest_summary_op])
                    summary_writer.add_summary(summary_str, batch_idx)
                    vtrain_summary_writer.add_summary(
                        vtrain_summary_str, batch_idx)
                    vtest_summary_writer.add_summary(
                        vtest_summary_str, batch_idx)
                # Save the model checkpoint periodically.
                if global_step_v % (10000*FLAGS.num_gpus) == 0:
                    checkpoint_path = os.path.join(
                        FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=batch_idx)
            # Stop the threads
            coord.request_stop()
            # Wait for threads to stop
            coord.join(threads)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
