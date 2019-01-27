import functools
from keras import backend as K
import tensorflow as tf
    
#https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value

    return wrapper

tf_auc = as_keras_metric(tf.metrics.auc)
#tf_precision = as_keras_metric(tf.metrics.precision)
#tf_recall = as_keras_metric(tf.metrics.recall)


def binary_cut(y):
    cutoff = 0.5
    return (K.cast(K.greater(y, cutoff), dtype='float32'))


def true_positive(y_true, y_pred):
    y_pred_bin = binary_cut(y_pred)
    return K.sum(y_true * y_pred_bin)


def false_positive(y_true, y_pred):
    y_pred_bin = binary_cut(y_pred)
    y_true_inv = 1 - y_true
    return K.sum(y_true_inv * y_pred_bin)


def false_negative(y_true, y_pred):
    y_pred_bin = binary_cut(y_pred)
    y_pred_inv = 1 - y_pred_bin
    return K.sum(y_true * y_pred_inv)


def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp + K.epsilon())


def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn + K.epsilon())


def fscore(y_true, y_pred):
    pre = precision(y_true, y_pred)
    re = recall(y_true, y_pred)
    return 2 * pre * re / (pre + re + K.epsilon())