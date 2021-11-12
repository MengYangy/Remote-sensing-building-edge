import tensorflow as tf


class Recall(tf.keras.metrics.Metric):
    def __init__(self, name='recall', **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
        self.recall = self.add_weight(name='recall', initializer='zeros')
        self.total = self.add_weight(name='total',  initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        build_true = y_true[:,:,:,0:1]
        build_pred = y_pred
        self.total.assign_add(tf.shape(y_true)[0])
        TP, TN, FP, FN = self.tp_tn_fp_fn(build_true, build_pred)
        # tf.print(TP, TN, FP, FN)
        values = TP / (TP + FP)
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weights = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)

        self.recall.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.recall/self.total

    def reset_states(self):
        self.recall.assign(0.)

    def tp_tn_fp_fn(self, y_true, y_pred):
        y_pred = tf.where(y_pred>0.5, 1, 0)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        TP = tf.reduce_sum(tf.cast(y_true * y_pred, tf.int32))
        TN = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.int32))
        FP = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.int32))
        FN = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.int32))

        TP = tf.cast(TP, tf.float32)
        TN = tf.cast(TN, tf.float32)
        FP = tf.cast(FP, tf.float32)
        FN = tf.cast(FN, tf.float32)
        return TP, TN, FP, FN


def precision(y_true, y_pred):
    build_true = y_true[:,:,:,0:1]
    build_pred = tf.where(y_pred>0,1.,0.)
    TP, TN, FP, FN = tp_tn_fp_fn(build_true, build_pred)
    value = TP / (TP + FP)
    return value

def recall(y_true, y_pred):
    build_true = y_true[:,:,:,0:1]
    build_pred = tf.where(y_pred>0,1.,0.)
    TP, TN, FP, FN = tp_tn_fp_fn(build_true, build_pred)
    value = TP / (TP + FN)
    return value

def iou(y_true, y_pred):
    build_true = y_true[:,:,:,0:1]
    build_pred = tf.where(y_pred>0,1.,0.)
    TP, TN, FP, FN = tp_tn_fp_fn(build_true, build_pred)
    value = TP / (TP + FP + FN)
    return value

def tp_tn_fp_fn(y_true, y_pred):
    y_pred = tf.where(y_pred>0.5, 1, 0)
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)

    TP = tf.reduce_sum(tf.cast(y_true * y_pred, tf.int32))
    TN = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.int32))
    FP = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.int32))
    FN = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.int32))

    TP = tf.cast(TP, tf.float32)
    TN = tf.cast(TN, tf.float32)
    FP = tf.cast(FP, tf.float32)
    FN = tf.cast(FN, tf.float32)
    return TP, TN, FP, FN