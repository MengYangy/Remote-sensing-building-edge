import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K


class Edge_Loss_V1(tf.keras.losses.Loss):
    def __init__(self):
        super(Edge_Loss_V1, self).__init__()

    def call(self, y_true, y_pred):
        ed_true = tf.cast(y_true[:, :, :, 1:], tf.float32)
        ed_pred = tf.cast(y_pred, tf.float32)

        # 提取建筑物边缘，作为正样本
        build_edge_bin = tf.multiply(tf.where(ed_true < 10, 1.0, 0), tf.where(ed_true > 0, 1.0, 0))
        build_edge_true = tf.multiply(build_edge_bin, ed_true)
        edge_pixel_num_true = K.sum(build_edge_bin)
        edge_pixel_num_true = tf.cast(edge_pixel_num_true, tf.float32)

        # 提取背景边缘，作为负样本
        background_edge_bin = tf.where(ed_true > 10, 0.1, 0)  # 把负样本中的30，40，50 变为3，4，5
        background_edge_true = tf.multiply(background_edge_bin, ed_true)
        edge_pixel_num_false = K.sum(background_edge_bin) * 10.0
        edge_pixel_num_false = tf.cast(edge_pixel_num_false, tf.float32)

        # 正样本的权重 β， 负样本权重 1-β
        edge_pixel_num = edge_pixel_num_true + edge_pixel_num_false  # + 1.0
        beta = edge_pixel_num_true / edge_pixel_num

        edge_loss_arr = beta * tf.multiply(build_edge_true, K.log(ed_pred)) + \
                        (1.0 - beta) * tf.multiply(background_edge_true, K.log(1.0 - ed_pred + K.epsilon()))

        # edge_loss_arr = beta * tf.multiply(ed_true, K.log(ed_pred))
        # edge_loss = -tf.reduce_mean(edge_loss_arr)
        edge_loss = -tf.reduce_sum(edge_loss_arr) / edge_pixel_num
        return edge_loss


class Edge_Loss_V2(tf.keras.losses.Loss):
    def __init__(self):
        super(Edge_Loss_V2, self).__init__()

    def call(self, y_true, y_pred):
        ed_true = tf.cast(y_true[:, :, :, 1:], tf.float32)
        ed_pred = tf.cast(y_pred, tf.float32)

        # 提取建筑物边缘，作为正样本
        build_edge_bin = tf.multiply(tf.where(ed_true < 10, 1.0, 0.0), tf.where(ed_true > 0, 1.0, 0))
        build_edge_true = tf.multiply(build_edge_bin, ed_true)
        edge_pixel_num_true = K.sum(build_edge_bin)
        edge_pixel_num_true = tf.cast(edge_pixel_num_true, tf.float32)

        # 提取背景边缘，作为负样本
        background_edge_bin = tf.where(ed_true > 10, 1.0, 0.0)  # 把负样本中的30，40，50 变为3，4，5
        background_edge_true = tf.multiply(background_edge_bin, tf.divide(ed_true, 10.0))
        edge_pixel_num_false = K.sum(background_edge_bin) * 10.0
        edge_pixel_num_false = tf.cast(edge_pixel_num_false, tf.float32)

        # 正样本的权重 β， 负样本权重 1-β
        edge_pixel_num = edge_pixel_num_true + edge_pixel_num_false  + 1.0
        beta = edge_pixel_num_true / edge_pixel_num


        # edge loss
        '''
        -y*log(1/(1+e^(-x))) = y*log(1+e^(-x))  当x为较大的负数时，损失会出现Nan，对此进一步化简
        y*log(1+e^(-x)) = y*log(1+e^(-x))+x*y - x*y = y*log(1-e^x) - x*y  
        合并：
        y*log(1+e^(-|x|)) - min(x, 0)
        '''
        build_loss = tf.subtract(
            tf.multiply(build_edge_bin,
                        tf.math.log1p(tf.math.exp(-tf.math.abs(ed_pred)))),
            tf.minimum(ed_pred, 0)
        )
        build_loss = tf.multiply(tf.cast(build_edge_true, tf.float32), build_loss)


        background_loss = tf.add(
            tf.multiply(background_edge_bin,
                        tf.math.log1p(tf.math.exp(-tf.math.abs(ed_pred)))),
            tf.maximum(ed_pred, 0)
        )
        background_loss = tf.multiply(tf.cast(background_edge_true, tf.float32), background_loss)

        '''
        loss = β * build_edge_loss / edge_pixel_num_true 
        + (1-β) * background_edge_loss / edge_pixel_num_false
        '''
        edge_loss = tf.add(tf.multiply(beta,
                                tf.divide(tf.reduce_sum(build_loss), edge_pixel_num_true)),
                    tf.multiply(tf.subtract(1.0, beta),
                                tf.divide(tf.reduce_sum(background_loss),
                                          edge_pixel_num_false)))
        # edge_loss = tf.add(tf.multiply(beta, build_loss),
        #                    tf.multiply((1.0 - beta), background_loss))

        # tf.print(tf.reshape(background_loss, (128,128)), summarize=-1)
        return edge_loss


# 结合自带的交叉熵
class Edge_Loss_V3(tf.keras.losses.Loss):
    def __init__(self):
        super(Edge_Loss_V3, self).__init__()

    def call(self, y_true, y_pred):
        ed_true = tf.cast(y_true[:, :, :, 1:], tf.float32)
        ed_pred = tf.cast(y_pred, tf.float32)

        # 提取建筑物边缘，作为正样本
        build_edge_bin = tf.multiply(tf.where(ed_true < 10, 1.0, 0), tf.where(ed_true > 0, 1.0, 0))
        build_edge_true = tf.multiply(build_edge_bin, ed_true)
        edge_pixel_num_true = K.sum(build_edge_bin)
        edge_pixel_num_true = tf.cast(edge_pixel_num_true, tf.float32)

        # 提取背景边缘，作为负样本
        background_edge_bin = tf.where(ed_true > 10, 0.1, 0)  # 把负样本中的30，40，50 变为3，4，5
        background_edge_true = tf.multiply(background_edge_bin, ed_true)
        edge_pixel_num_false = K.sum(background_edge_bin) * 10.0
        edge_pixel_num_false = tf.cast(edge_pixel_num_false, tf.float32)
        # 正样本的权重 β， 负样本权重 1-β
        edge_pixel_num = edge_pixel_num_true + edge_pixel_num_false + 1.0
        beta = edge_pixel_num_true / edge_pixel_num

        print(build_edge_true)
        print(build_edge_bin)
        print(ed_pred)
        print(background_edge_bin)
        print(background_edge_true)
        # edge loss
        build_loss = tf.multiply(
            build_edge_true,
            binary_crossentropy(y_true=build_edge_bin, y_pred=ed_pred)
        )

        background_loss = tf.multiply(
            background_edge_true,
            binary_crossentropy(y_true=tf.where(background_edge_bin > 0, 1., 0.),
                                y_pred=tf.subtract(1.0, ed_pred))
        )

        '''
        loss = β * build_edge_loss / edge_pixel_num_true 
        + (1-β) * background_edge_loss / edge_pixel_num_false
        '''
        edge_loss = tf.add(tf.multiply(beta,
                                tf.divide(tf.reduce_sum(build_loss), edge_pixel_num_true)),
                    tf.multiply(tf.subtract(1.0, beta),
                                tf.divide(tf.reduce_sum(background_loss),
                                          edge_pixel_num_false)))
        return edge_loss


class Build_Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(Build_Loss, self).__init__()

    def call(self, y_true, y_pred):
        build_true = y_true[:,:,:,0:1]
        build_pred = y_pred
        # tf.print(build_true.shape, build_true)
        # tf.print(y_pred.shape, y_pred)
        loss = tf.maximum(build_pred, 0) - \
               tf.multiply(build_true, build_pred) + \
               tf.math.log1p(tf.math.exp(-tf.math.abs(build_pred)))
        # loss = tf.keras.losses.binary_crossentropy(build_true, build_pred)
        loss = tf.reduce_mean(loss)

        return loss


