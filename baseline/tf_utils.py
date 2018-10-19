""" 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Confusion_Matrix(object):
    """
    """

    def __init__(self, logits, label):
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(logits * label)
        tn = tf.count_nonzero((logits - 1) * (label - 1))
        fp = tf.count_nonzero(logits * (label - 1))
        fn = tf.count_nonzero((logits - 1) * label)

        # Calculate accuracy, precision, recall and F1 score.
        self.accuracy = (tp + tn) / tf.maximum(tp + fp + fn + tn, 1)
        self.precision = tp / tf.maximum(tp + fp, 1)
        self.recall = tp / tf.maximum(tp + fn, 1)
        self.f1_score = (2 * self.precision * self.recall) / tf.maximum(
            self.precision + self.recall, 1e-8)
        self.f2_score = (
            1.0 + 2.0**2) * self.precision * self.recall / tf.maximum(
                2.0**2 * self.precision + self.recall, 1e-8)
