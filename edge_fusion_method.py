import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization, \
    Add, Activation, Concatenate, Multiply, Input
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.utils import plot_model


class Edge_Fusion_Method():
    def __init__(self, filters=120):
        self.filters = filters
        self.split_filters = 40
        self.height_weight = 0.5
        self.middle_weight = 0.3
        self.low_weight = 0.2

    def conv(self, input_tensor, filters=120, s=1, is1x1=False):
        if is1x1:
            out = Conv2D(filters=filters,
                         kernel_size=(1, 1))(input_tensor)
        else:
            out = Conv2D(filters=filters,
                         kernel_size=(3, 3),
                         strides=s,
                         padding='same')(input_tensor)
        out = BatchNormalization()(out)
        out = self.leaky_relu(out)
        return out

    def res_block(self, input_tensor, f_size, s=1):
        out = Conv2D(filters=f_size,
                     kernel_size=(3, 3),
                     strides=s,
                     padding='same')(input_tensor)
        out = BatchNormalization()(out)
        out = self.leaky_relu(out)
        out = Add()([input_tensor, out])
        return out

    def leaky_relu(self, input_tensor):
        out = tf.nn.leaky_relu(input_tensor, alpha=0.1)
        return out

    def fusion_to_height(self, conv1_1, conv2_1, conv3_1):
        # print(conv2_1.shape, conv3_1.shape)
        conv2_1 = self.up_samp(conv3_1, conv2_1)
        conv1_1 = self.up_samp(conv2_1, conv1_1)
        conv_height = self.res_block(conv1_1, f_size=self.split_filters)
        return conv_height

    def fusion_to_middle(self, conv1_2, conv2_2, conv3_2):
        conv2_2 = self.up_samp(conv3_2, conv2_2)
        conv2_2 = self.down_samp(conv1_2, conv2_2)
        conv_middle = self.res_block(conv2_2, f_size=self.split_filters)
        return conv_middle

    def fusion_to_low(self, conv1_3, conv2_3, conv3_3):
        conv2_3 = self.down_samp(conv1_3, conv2_3)
        conv3_3 = self.down_samp(conv2_3, conv3_3)
        conv_low = self.res_block(conv3_3, f_size=self.split_filters)
        return conv_low

    def up_samp(self, tensor1_need_up, tensor2):
        tensor1 = Conv2DTranspose(filters=self.split_filters, kernel_size=(3, 3),
                                  strides=2, padding='same')(tensor1_need_up)
        tensor1 = self.leaky_relu(tensor1)
        # tensor2 = Concatenate()([tensor2, tensor1])
        tensor2 = Add()([tensor2, tensor1])
        tensor2 = self.res_block(tensor2,  f_size=self.split_filters)
        return tensor2

    def down_samp(self, tensor1_need_down, tensor2):
        tensor1_need_down = self.conv(tensor1_need_down, filters=self.split_filters, s=2)
        # tensor2 = Concatenate()([tensor2, tensor1_need_down])
        tensor2 = Add()([tensor2, tensor1_need_down])
        tensor2 = self.res_block(tensor2, f_size=self.split_filters)
        return tensor2

    def pre_procession(self, input_tensor1, input_tensor2, input_tensor3):
        input_tensor1 = self.conv(input_tensor1, filters=120, is1x1=True)
        input_tensor2 = self.conv(input_tensor2, filters=120, is1x1=True)
        input_tensor3 = self.conv(input_tensor3, filters=120, is1x1=True)
        for i in range(2):
            input_tensor1 = self.res_block(input_tensor1, f_size=120)
            input_tensor2 = self.res_block(input_tensor2, f_size=120)
            input_tensor3 = self.res_block(input_tensor3, f_size=120)
        return input_tensor1, input_tensor2, input_tensor3

    def add_weight_edge(self, conv_height, conv_middle, conv_low):
        '''
            out = 0.5 * conv_height + 0.3 * conv_middle + 0.2 * conv_low
        '''
        conv_height = self.conv(conv_height, filters=1, is1x1=True)
        conv_middle = self.conv(conv_middle, filters=1, is1x1=True)
        conv_low = self.conv(conv_low, filters=1, is1x1=True)
        conv_height = tf.multiply(conv_height, self.height_weight)
        conv_middle = tf.multiply(conv_middle, self.middle_weight)
        conv_low = tf.multiply(conv_low, self.low_weight)
        return Add()([conv_height, conv_middle, conv_low])

    def call(self, input_tensor1, input_tensor2, input_tensor3):
        '''
        :param input_tensor1: 第一级分辨率特征图
        :param input_tensor2: 第二级分辨率特征图
        :param input_tensor3: 第三级分辨率特征图
        :return:
        '''
        conv1, conv2, conv3 = self.pre_procession(input_tensor1, input_tensor2, input_tensor3)
        conv1_1, conv1_2, conv1_3= tf.split(conv1, num_or_size_splits=3, axis=3)
        conv2_1, conv2_2, conv2_3= tf.split(conv2, num_or_size_splits=3, axis=3)
        conv3_1, conv3_2, conv3_3= tf.split(conv3, num_or_size_splits=3, axis=3)

        conv_height = self.fusion_to_height(conv1_1, conv2_1, conv3_1)
        conv_middle = self.fusion_to_middle(conv1_2, conv2_2, conv3_2)
        conv_low = self.fusion_to_low(conv1_3, conv2_3, conv3_3)

        conv_low = Conv2DTranspose(filters=self.split_filters, kernel_size=3, strides=4, padding='same')(conv_low)
        conv_low = self.leaky_relu(conv_low)

        conv_middle = Conv2DTranspose(filters=self.split_filters, kernel_size=3, strides=2, padding='same')(conv_middle)
        conv_middle = self.leaky_relu(conv_middle)
        # build_pred
        out_tensor = Concatenate()([conv_low, conv_middle, conv_height])
        build_tensor = self.conv(out_tensor, filters=64)
        # edge_pred

        edge_tensor = self.add_weight_edge(conv_height, conv_middle, conv_low)

        return build_tensor, edge_tensor


if __name__ == '__main__':
    input1 = Input(shape=(512,512,3))
    input2 = Input(shape=(256,256,3))
    input3 = Input(shape=(128,128,3))
    edge_fusion_ = Edge_Fusion_Method()
    build_tensor, edge_tensor = edge_fusion_.call(input1, input2, input3)
    model = Model([input1, input2, input3], build_tensor)
    model.summary()
    plot_model(model, './model.png')

