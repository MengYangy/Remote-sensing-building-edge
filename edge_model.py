import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPool2D, \
    Input, Add, Activation, Conv2DTranspose, UpSampling2D

from tensorflow.keras.models import Model
from config import *
from backbone.resNet import ResNetFamily
from backbone.resnext import ResNeXt

from edge_fusion_method import Edge_Fusion_Method
from uility.activate import leak_relu


def conv_bn_relu(input_tensor, f_size, k_size=3, s=1):
    x1 = Conv2D(f_size, kernel_size=(k_size,k_size),strides=s, padding='same')(input_tensor)
    bn1 = BatchNormalization()(x1)
    relu1 = leak_relu(bn1)
    return relu1

def block(input_tensor):
    pass

def encode(input_tensor):
    conv1 = conv_bn_relu(input_tensor, f_size=Filter_Size, k_size=3)
    conv1 = conv_bn_relu(conv1, f_size=Filter_Size, k_size=3)

    conv2 = conv_bn_relu(conv1, f_size=Filter_Size, k_size=3, s=2)
    conv2 = conv_bn_relu(conv2, f_size=2*Filter_Size)
    conv2 = conv_bn_relu(conv2, f_size=2*Filter_Size)

    conv3 = conv_bn_relu(conv2, f_size=2 * Filter_Size, k_size=3, s=2)
    conv3 = conv_bn_relu(conv3, f_size=4 * Filter_Size)
    conv3 = conv_bn_relu(conv3, f_size=4 * Filter_Size)

    conv4 = conv_bn_relu(conv3, f_size=4 * Filter_Size, k_size=3, s=2)
    conv4 = conv_bn_relu(conv4, f_size=8 * Filter_Size)
    conv4 = conv_bn_relu(conv4, f_size=8 * Filter_Size)

    conv5 = conv_bn_relu(conv4, f_size=8 * Filter_Size, k_size=3, s=2)
    conv5 = conv_bn_relu(conv5, f_size=8 * Filter_Size)
    conv5 = conv_bn_relu(conv5, f_size=8 * Filter_Size)

    return conv1,conv2,conv3,conv4,conv5

def transpose(input_tensor, f_size):
    out = Conv2DTranspose(f_size,
                          kernel_size=3,
                          strides=2,
                          padding='same',
                          activation='relu')(input_tensor)
    return out


def short_concat_conv(input_tensor1, input_tensor2, f_size):
    input_tensor2 = transpose(input_tensor2, f_size)
    conv = tf.concat([input_tensor1, input_tensor2], axis=-1)
    conv = conv_bn_relu(conv, f_size, k_size=1)
    conv = res_block(conv, f_size)
    # conv = conv_bn_relu(conv, f_size)
    return conv

def res_block(input_tensor, f_size):
    out = Conv2D(filters=f_size,
                 kernel_size=(3, 3),
                 padding='same')(input_tensor)
    out = BatchNormalization()(out)
    out = leak_relu(out)
    out = Add()([input_tensor, out])
    return out


def decode(conv1,conv2,conv3,conv4,conv5):
    conv4_1 = short_concat_conv(conv4, conv5, f_size=Filter_Size*8)
    conv3_1 = short_concat_conv(conv3, conv4_1, f_size=Filter_Size*4)
    conv2_1 = short_concat_conv(conv2, conv3_1, f_size=Filter_Size*2)
    conv1_1 = short_concat_conv(conv1, conv2_1, f_size=Filter_Size)

    return conv1_1, conv2_1, conv3_1


def down_sample(input_tensor, f_size, k_size=3, s=2):
    out = Conv2D(filters=f_size, kernel_size=k_size, strides=s, padding='same', activation='relu')(input_tensor)
    return out

def pred_layer(input_tensor, name=''):
    out = Conv2D(filters=1, kernel_size=3, padding='same', name=name)(input_tensor)
    return out

def up(input_tensor, f_size):
    out = Conv2DTranspose(f_size,kernel_size=3, strides=2, padding='same', activation='relu')(input_tensor)
    return out


def short_concat_conv1(input_tensor1, input_tensor2, f_size):
    conv = tf.concat([input_tensor1, input_tensor2], axis=-1)
    conv = conv_bn_relu(conv, f_size, k_size=1)
    conv = conv_bn_relu(conv, f_size)
    conv = conv_bn_relu(conv, f_size)
    return conv


def edge_feature(conv1_1, conv2_1, conv3_1):
    conv1_1_down = down_sample(conv1_1, f_size=Filter_Size)
    conv1_1_1 = conv_bn_relu(conv1_1, f_size=Filter_Size)
    conv1_1_1 = conv_bn_relu(conv1_1_1, f_size=Filter_Size)

    conv2_1_down = down_sample(tf.concat([conv2_1, conv1_1_down], axis=-1), f_size=2*Filter_Size)
    conv2_1_1 = short_concat_conv1(conv1_1_down, conv2_1, f_size=2*Filter_Size)

    conv3_1_1 = short_concat_conv1(conv2_1_down, conv3_1, f_size=4 * Filter_Size)
    conv3_1_1_up = up(conv3_1_1, f_size=2*Filter_Size)
    conv2_1_1 = tf.concat([conv2_1_1, conv3_1_1_up], axis=-1)
    conv2_1_1 = Conv2D(Filter_Size*2, 1)(conv2_1_1)

    conv2_1_1_up = up(conv2_1_1, f_size=Filter_Size)
    conv1_1_1 = tf.concat([conv1_1_1, conv2_1_1_up], axis=-1)
    conv1_1_1 = Conv2D(Filter_Size, 1)(conv1_1_1)

    # ??????????????????    ?????????????????????????????????????????????
    pred = conv1_1_1
    edge_pred3 = pred_layer(UpSampling2D((4,4))(conv3_1_1))
    edge_pred2 = pred_layer(UpSampling2D()(conv2_1_1))
    edge_pred1 = pred_layer(conv1_1_1)

    return pred, edge_pred1, edge_pred2, edge_pred3


def edge_feature1(conv1_1, conv2_1, conv3_1):
    def down_fusion(conv1_1, conv2_1, conv3_1):
        c1_down = Conv2D(2*Filter_Size, 3,strides=2 ,padding='same', activation='relu')(conv1_1)
        c1_down = BatchNormalization()(c1_down)

        conv2_1 = Add()([c1_down, conv2_1])
        c2_down = Conv2D(4 * Filter_Size, 3, strides=2, padding='same', activation='relu')(conv2_1)
        c2_down = BatchNormalization()(c2_down)

        conv3_1 = Add()([c2_down, conv3_1])
        return conv1_1, conv2_1, conv3_1

    def up_fusion(conv1_1, conv2_1, conv3_1):
        c3_up = Conv2DTranspose(filters=Filter_Size*2, kernel_size=3, strides=2, padding='same',
                                activation='relu')(conv3_1)
        c3_up = BatchNormalization()(c3_up)

        conv2_1 = Add()([c3_up, conv2_1])
        conv2_1 = conv_bn_relu(conv2_1, f_size=Filter_Size * 2)
        c2_up = Conv2DTranspose(filters=Filter_Size, kernel_size=3, strides=2, padding='same',
                                activation='relu')(conv2_1)
        c2_up = BatchNormalization()(c2_up)

        conv1_1 = Add()([c2_up, conv1_1])
        conv1_1 = conv_bn_relu(conv1_1, f_size=Filter_Size)
        return conv1_1, conv2_1, conv3_1

    conv1_1, conv2_1, conv3_1 = down_fusion(conv1_1, conv2_1, conv3_1)
    conv1_1 = conv_bn_relu(conv1_1, f_size=Filter_Size)
    conv1_1 = conv_bn_relu(conv1_1, f_size=Filter_Size)

    conv2_1 = conv_bn_relu(conv2_1, f_size=Filter_Size * 2)
    conv2_1 = conv_bn_relu(conv2_1, f_size=Filter_Size * 2)

    conv3_1 = conv_bn_relu(conv3_1, f_size=Filter_Size * 4)
    conv3_1 = conv_bn_relu(conv3_1, f_size=Filter_Size * 4)

    conv1_1, conv2_1, conv3_1 = up_fusion(conv1_1, conv2_1, conv3_1)

    pred = conv1_1
    edge_pred1 = conv1_1
    edge_pred3 = conv_bn_relu(UpSampling2D((4, 4))(conv3_1), f_size=2*Filter_Size)
    edge_pred3 = conv_bn_relu(edge_pred3, f_size=Filter_Size)

    edge_pred2 = conv_bn_relu(UpSampling2D()(conv2_1), f_size=Filter_Size)

    return pred, edge_pred1, edge_pred2, edge_pred3


def pred_module(build_pred, edge_pred):
    build_pred = Conv2D(1, 3, padding='same', name='build_pred')(build_pred)
    edge_pred = Conv2D(1, 3, padding='same', name='edge_pred')(edge_pred)
    return build_pred, edge_pred


def myModel(backbone='resnet'):
    input_tensor = Input(shape=(Image_Size,Image_Size,3))

    # ??????????????????
    # conv1,conv2,conv3,conv4,conv5 = encode(input_tensor)
    if backbone == 'resnet':
        resnet = ResNetFamily(f_size=64, layer_num=50)
        conv1, conv2, conv3, conv4, conv5 = resnet.resnet_n(input_tensor)
    elif backbone == 'resnext50':
        resnext = ResNeXt(layers_num=50, f_size=Filter_Size, cardinality=8)
        conv1, conv2, conv3, conv4, conv5 = resnext.call(input_tensor)
    else:
        raise ValueError('Not find this backbone!')


    # ????????????
    conv1_1, conv2_1, conv3_1 = decode(conv1,conv2,conv3,conv4,conv5)

    # ??????????????????????????????
    edge_fusion_ = Edge_Fusion_Method()
    build_tensor, edge_tensor = edge_fusion_.call(conv1_1, conv2_1, conv3_1)

    build_pred = tf.concat([conv1_1, build_tensor], axis=-1)
    build_pred = conv_bn_relu(build_pred, Filter_Size)
    build_pred, edge_pred = pred_module(build_pred, edge_tensor)
    model = Model(input_tensor, [build_pred, edge_pred])
    return model


    # pred, edge_pred1, edge_pred2, edge_pred3 = edge_feature1(conv1_1, conv2_1, conv3_1)
    #
    # # ????????????
    # build_pred = tf.concat([conv1_1, pred], axis=-1)
    # build_pred = conv_bn_relu(build_pred, Filter_Size)
    # build_pred, edge_pred1, edge_pred2, edge_pred3 = pred_module(build_pred, edge_pred1, edge_pred2, edge_pred3)
    #
    # # finally_pred = tf.concat([build_pred, edge_pred1, edge_pred2, edge_pred3], axis=-1)
    # # out_tensor = finally_pred
    #
    # return Model(input_tensor, [build_pred, edge_pred1, edge_pred2, edge_pred3])
