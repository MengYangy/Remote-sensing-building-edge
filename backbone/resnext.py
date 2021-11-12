import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, MaxPool2D, Input, \
    GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.models import Model
from uility.activate import leak_relu


Activate = leak_relu

'''
残差分组卷积
input_tensor：输入数据
filter_size：滤波器数量
num_or_size_splits：分组数量32组，
每组卷积滤波器数量filter_x为filter_size/32
'''
def identity_block(input_tensor, filter_size):
    input_tensor_list = tf.split(input_tensor, num_or_size_splits=32, axis=3)
    filter_x = filter_size//32
    def conv_fun(input_tensor):
        convx = Conv2D(filter_x, 1, padding='same')(input_tensor)
        convx = BatchNormalization()(convx)
        convx = Activate(convx)
        convx = Conv2D(filter_x, 3, padding='same')(convx)
        convx = BatchNormalization()(convx)
        convx = Activate(convx)
        return convx
    out_list = list(map(conv_fun,input_tensor_list))
    out = tf.concat(out_list, axis=-1)
    out = Conv2D(filter_size, 1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Add()([input_tensor, out])
    out = Activate(out)
    return out


def conv_bn(filter_size,x):
    convx = Conv2D(filter_size, 3, padding='same')(x)
    bnx = BatchNormalization()(convx)
    acx = Activate(bnx)
    return acx


def ResNeXt_50(input_shape=(512, 512, 3), filters=64):
    inputs = Input(input_shape)
    '''
    第一层
    '''
    conv1 = conv_bn(filters, inputs)
    conv1 = conv_bn(filters, conv1)
    conv1 = conv_bn(filters, conv1)
    pool1 = MaxPool2D()(conv1)

    '''
    第二层
    '''
    conv2 = conv_bn(filters * 2, pool1)
    for _ in range(3):
        conv2 = identity_block(conv2, filters * 2)
    pool2 = MaxPool2D()(conv2)

    '''
    第三层
    '''
    conv3 = conv_bn(filters * 4, pool2)
    for _ in range(4):
        conv3 = identity_block(conv3, filters * 4)
    pool3 = MaxPool2D()(conv3)

    '''
    第四层
    '''
    conv4 = conv_bn(filters * 8, pool3)
    for _ in range(6):
        conv4 = identity_block(conv4, filters * 8)
    pool4 = MaxPool2D()(conv4)

    '''
    第五层
    '''
    conv5 = conv_bn(filters * 16, pool4)
    for _ in range(3):
        conv5 = identity_block(conv5, filters * 16)

    x = GlobalAveragePooling2D()(conv5)
    x = Dense(100, activation='softmax')(x)

    model = Model(inputs, x)
    return model


class ResNeXt():
    '''
    ResNeXt
    Arguments:
        f_size=64,  初始输入时卷积层中滤波器的数量，默认为64，随后依次成倍增加
        cardinality=32,     分组数量，默认32组
        input_shape=(512,512,3),    输入数据尺寸
        layers_num=50       ResNeXt，默认50代表ResNeXt50，101表示ResNeXt101，其他数字也表示ResNeXt101，—.—
    '''
    def __init__(self, f_size=64, cardinality=32, input_shape=(512,512,3), layers_num=50):
        self.Filter_size = f_size
        self.cardinality = cardinality
        self.input_shape = input_shape
        self.layers_num = layers_num

    def resNext_block(self, input_tensor, f_size, block_type='b'):
        '''
        提供了三种block，
        a表示使用1x1卷积把输入分成32组，最后使用Add输出
        b表示使用1x1卷积把输入分成32组，最后使用 Concatenate 输出
        其他的表示使用 split 把输入分成32组，最后使用 Concatenate 输出
        :param input_tensor:输入的tensor
        :param f_size:滤波器数量
        :param block_type:选择 block 类型
        :return:tensor
        '''
        def block_a(input_tensor, f_size):
            conv1 = self.conv(input_tensor, f_size=f_size // 64, k_size=1)
            conv2 = self.conv(conv1, f_size=f_size // 64, k_size=3)
            conv3 = self.conv(conv2, f_size=f_size, k_size=1)
            return conv3

        def block_b(input_tensor, f_size):
            conv1 =self.conv(input_tensor, f_size=f_size//64, k_size=1)
            conv2 =self.conv(conv1, f_size=f_size//64, k_size=3)
            return conv2

        if block_type == 'a':   # 原论文中的a型结构
            finally_conv = [block_a(input_tensor, f_size) for _ in range(self.cardinality)]
            finally_conv = Add()(finally_conv)

        elif block_type == 'b': # 原论文中的b型结构
            finally_conv = [block_b(input_tensor, f_size) for _ in range(self.cardinality)]
            finally_conv = Concatenate()(finally_conv)
            finally_conv = self.conv(finally_conv, f_size=f_size, k_size=1)

        else:   # 原论文中，用split替换1x1卷积的b型结构
            input_tensor_list = tf.split(input_tensor, num_or_size_splits=self.cardinality, axis=3)
            finally_conv = list(map(block_b, input_tensor_list))
            finally_conv = Concatenate()(finally_conv)
            finally_conv = self.conv(finally_conv, f_size=f_size, k_size=1)
        out = Add()([finally_conv, input_tensor])
        return out


    def conv(self, input_tensor, f_size, k_size=3, s=1):
        'Conv2D + BN + Leak_relu'
        if k_size == 3:
            out = Conv2D(filters=f_size, kernel_size=k_size, strides=s, padding='same')(input_tensor)
        else:
            out = Conv2D(filters=f_size, kernel_size=k_size, strides=s)(input_tensor)
        out = BatchNormalization()(out)
        out = leak_relu(out)
        return out

    def call(self, input_tensor):
        if self.layers_num == 50:
            layers_num = [3, 4, 6, 3]
        else:
            layers_num = [3, 4, 23, 3]
        # input_tensor = Input(shape=(self.input_shape))
        'The first level'
        conv1 = self.conv(input_tensor, f_size=self.Filter_size, k_size=3)
        conv1 = self.conv(conv1, f_size=self.Filter_size, k_size=3)
        conv1 = self.conv(conv1, f_size=self.Filter_size, k_size=3)

        'The second level'
        conv2 = MaxPool2D()(conv1)
        for i in range(layers_num[0]):
            conv2 = self.conv(conv2, f_size=self.Filter_size*2, k_size=1)
            conv2 = self.resNext_block(conv2, f_size=self.Filter_size*2)
            conv2 = self.conv(conv2, f_size=self.Filter_size*4, k_size=1)

        'The third level'
        conv3 = MaxPool2D()(conv2)
        for i in range(layers_num[1]):
            conv3 = self.conv(conv3, f_size=self.Filter_size * 4, k_size=1)
            conv3 = self.resNext_block(conv3, f_size=self.Filter_size * 4)
            conv3 = self.conv(conv3, f_size=self.Filter_size * 8, k_size=1)

        'The fourth level'
        conv4 = MaxPool2D()(conv3)
        for i in range(layers_num[2]):
            conv4 = self.conv(conv4, f_size=self.Filter_size * 8, k_size=1)
            conv4 = self.resNext_block(conv4, f_size=self.Filter_size * 8)
            conv4 = self.conv(conv4, f_size=self.Filter_size * 16, k_size=1)

        'The fifth level'
        conv5 = MaxPool2D()(conv4)
        for i in range(layers_num[3]):
            conv5 = self.conv(conv5, f_size=self.Filter_size * 16, k_size=1)
            conv5 = self.resNext_block(conv5, f_size=self.Filter_size * 16)
            conv5 = self.conv(conv5, f_size=self.Filter_size * 32, k_size=1)

        # x = GlobalAveragePooling2D()(conv5)
        # x = Dense(1000, activation='softmax')(x)
        #
        # model = Model(input_tensor, x)
        # return model
        return conv1, conv2, conv3, conv4, conv5


# if __name__ == '__main__':
#     resnext = ResNeXt(layers_num=50)
#     model = resnext.call()
#     model.summary()
