import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization, \
    Input, Activation, Add
from tensorflow.keras.models import Model
from uility.activate import leak_relu
from edge_fusion_method import Edge_Fusion_Method

class ResNetFamily():
    '''
    ResNetFamily:ResNet家族
    两个参数
        f_size：初始输入时卷积层滤波器的数量
    '''
    def __init__(self, f_size=64, layer_num=50):
        self.f_size = f_size
        self.layer_num = layer_num

    def activate(self, input_tensor):
        out = Activation('relu')(input_tensor)
        return out

    def conv_n(self, input_tensor, f_size, s=1, is_1x1=False):
        if is_1x1:
            conv = Conv2D(filters=f_size, kernel_size=(1, 1), strides=s)(input_tensor)
        else:
            conv = Conv2D(filters=f_size, kernel_size=(3,3), padding='same', strides=s)(input_tensor)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        return conv

    def res_block(self, input_tensor, f_size):
        conv1 = self.conv_n(input_tensor, f_size=f_size, is_1x1=True)
        conv2 = self.conv_n(conv1, f_size=f_size)
        conv3 = self.conv_n(conv2, f_size=f_size*4, is_1x1=True)
        out = Add()([input_tensor, conv3])
        return out

    def resnet_n(self, input_tensor):
        f_sizes = [self.f_size, self.f_size, self.f_size*2, self.f_size*4, self.f_size*8]
        if self.layer_num == 50:
            nums= [3, 4, 6, 3]
        elif self.layer_num == 101:
            nums = [3, 4, 23, 3]
        elif self.layer_num == 152:
            nums = [3, 8, 36, 3]
        else:
            nums= [3, 4, 6, 3]
        # 第一层
        conv1 = self.conv_n(input_tensor, f_size=f_sizes[0])
        conv1 = self.conv_n(conv1, f_size=f_sizes[0])
        conv1 = self.conv_n(conv1, f_size=f_sizes[0])

        # 第二层
        conv2 = self.conv_n(conv1, f_size=int(f_sizes[1])*4, s=2, is_1x1=True)
        for i in range(nums[0]):
            conv2 = self.res_block(conv2, f_size=f_sizes[1])

        # 第三层
        conv3 = self.conv_n(conv2, f_size=int(f_sizes[2]) * 4, s=2, is_1x1=True)
        for i in range(nums[1]):
            conv3 = self.res_block(conv3, f_size=f_sizes[2])

        # 第四层
        conv4 = self.conv_n(conv3, f_size=int(f_sizes[3]) * 4, s=2, is_1x1=True)
        for i in range(nums[2]):
            conv4 = self.res_block(conv4, f_size=f_sizes[3])

        # 第五层
        conv5 = self.conv_n(conv4, f_size=int(f_sizes[4]) * 4, s=2, is_1x1=True)
        for i in range(nums[3]):
            conv5 = self.res_block(conv5, f_size=f_sizes[4])

        # conv1 = Conv2D(64, kernel_size=1, name='Conv1', activation='relu')(conv1)
        conv2 = Conv2D(128, kernel_size=1, name='Conv2', activation='relu')(conv2)
        conv3 = Conv2D(256, kernel_size=1, name='Conv3', activation='relu')(conv3)
        conv4 = Conv2D(512, kernel_size=1, name='Conv4', activation='relu')(conv4)
        conv5 = Conv2D(1024, kernel_size=1, name='Conv5', activation='relu')(conv5) # 2098176
        return conv1, conv2, conv3, conv4, conv5

if __name__ == '__main__':
    input_tensor = Input((512,512,3))
    resnet = ResNetFamily(f_size=64, layer_num=50)
    conv1, conv2, conv3, conv4, conv5 = resnet.resnet_n(input_tensor)
    model = Model(input_tensor, [conv1, conv2, conv3, conv4, conv5])
    model.summary()
    print(model.output)

