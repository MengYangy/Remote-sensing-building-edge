import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from config import *
import numpy as np
import math


IMG_SIZE = Image_Size

def read_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img / 127.5 - 1
    return img

def read_lab(path, is_edge_lab=False):
    lab = tf.io.read_file(path)
    lab = tf.image.decode_png(lab, channels=1)
    if not is_edge_lab:
        lab /= 255
    return lab

def load_dataset(input_img_path, input_lab_path, input_edge_lab_path):
    img = read_img(input_img_path)
    lab = read_lab(input_lab_path)
    edge_lab = read_lab(input_edge_lab_path, is_edge_lab=True)
    img = tf.image.resize(img, [IMG_SIZE,IMG_SIZE])
    lab = tf.image.resize(lab, [IMG_SIZE,IMG_SIZE])
    edge_lab = tf.image.resize(edge_lab, [IMG_SIZE,IMG_SIZE])
    all_lab = tf.concat([lab, edge_lab], axis=-1)
    return img, all_lab

def load_val(input_img_path, input_lab_path):
    img = read_img(input_img_path)
    lab = read_lab(input_lab_path)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    lab = tf.image.resize(lab, [IMG_SIZE, IMG_SIZE])
    return img, lab

def load_train_func():
    dataset_file_paths = r'D:\dataset\newedge'
    dataset_img = tf.io.gfile.glob(dataset_file_paths + '/image/*')
    dataset_lab = tf.io.gfile.glob(dataset_file_paths + '/label/*')
    dataset_edge_lab = tf.io.gfile.glob(dataset_file_paths + '/edgelabel/*')

    dataset_img.sort()
    dataset_lab.sort()
    dataset_edge_lab.sort()


    BATCH_SIZE = Batch_Size
    train_step = len(dataset_img) // BATCH_SIZE
    BUFFER_SIZE = BUFFER_Size
    # STEPS_PER_EPOCH = len(dataset_img) // BATCH_SIZE
    AUTO = tf.data.experimental.AUTOTUNE

    dataset = tf.data.Dataset.from_tensor_slices((dataset_img, dataset_lab, dataset_edge_lab))
    train_dataset = dataset.map(load_dataset, num_parallel_calls=AUTO)
    train_dataset = train_dataset.cache().repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)
    return train_dataset, train_step


def load_val_func():
    val_file_paths = r'D:\dataset\newedge'
    val_img = tf.io.gfile.glob(val_file_paths + '/image/*')
    val_lab = tf.io.gfile.glob(val_file_paths + '/label/*')

    val_img.sort()
    val_lab.sort()
    BATCH_SIZE = Batch_Size
    val_step = len(val_lab) // BATCH_SIZE
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((val_img, val_lab))
    val_dataset = dataset.map(load_val, num_parallel_calls=AUTO)
    val_dataset = val_dataset.cache().batch(BATCH_SIZE)
    return val_dataset, val_step


# 使用tf.keras.utils.Sequence创建数据生成器
class Load_data(tf.keras.utils.Sequence):
    def __init__(self, dataset_type):
        super(Load_data, self).__init__()

        self.img_path = IMAGE_PATH if dataset_type == 'train' else VAL_IMAGE_PATH
        self.lab_path = LABEL_PATH if dataset_type == 'train' else VAL_LABEL_PATH
        self.edge_lab_path = EDGE_LABEL_PATH if dataset_type == 'train' else VAL_EDGELABEL_PATH

        self.dataset_type = dataset_type
        self.batch_size = Batch_Size
        self.image_names = np.array(os.listdir(self.img_path))

        self.img_nums = len(self.image_names)
        self.soft_list = np.arange(0, self.img_nums)
        np.random.shuffle(self.image_names)
        self.img_arr = []
        self.lab_arr = []
        # self.load_data()

    def __len__(self):
        return math.ceil(self.img_nums / self.batch_size)

    def __getitem__(self, item):
        img_paths = self.image_names[self.batch_size * item : self.batch_size * (item + 1)]
        lab_paths = self.image_names[self.batch_size * item : self.batch_size * (item + 1)]
        len_ = len(self.image_names[self.batch_size * item : self.batch_size * (item + 1)])
        img_tensor, lab_tensor = self.load_data(img_paths, lab_paths, len_=len_)
        img_tensor, lab_tensor = tf.cast(img_tensor, tf.float32), tf.cast(lab_tensor, tf.float32)
        return img_tensor, lab_tensor

    def load_data(self, img_paths, lab_paths, len_=0):  # 自定义函数，读取数据、数据增强等 可在这里进行。
        self.img_tensor = np.zeros((len_, Image_Size, Image_Size, 3))


        for i in range(len_):
            img_data = self.read_img(os.path.join(self.img_path, img_paths[i]))
            self.img_tensor[i, :, :, :] = img_data


        if self.dataset_type == 'train':
            self.lab_tensor = np.zeros((len_, Image_Size, Image_Size, 2))
            for i in range(len_):
                lab_data = self.read_lab(os.path.join(self.lab_path, lab_paths[i]))
                self.lab_arr = np.zeros((Image_Size, Image_Size, 2))
                self.lab_arr[:, :, 0:1] = lab_data
                edge_lab_data = self.read_lab(os.path.join(self.edge_lab_path, lab_paths[i]), is_edge_lab=True)
                self.lab_arr[:, :, 1:2] = edge_lab_data
                self.lab_tensor[i, :, :, :] = self.lab_arr

        else:
            self.lab_tensor = np.zeros((len_, Image_Size, Image_Size, 2))
            for i in range(len_):
                lab_data = self.read_lab(os.path.join(self.lab_path, lab_paths[i]))
                self.lab_arr = np.zeros((Image_Size, Image_Size, 2))
                self.lab_arr[:, :, 0:1] = lab_data
                edge_lab_data = self.read_lab(os.path.join(self.edge_lab_path, lab_paths[i]), is_edge_lab=True)
                self.lab_arr[:, :, 1:2] = edge_lab_data
                self.lab_tensor[i, :, :, :] = self.lab_arr

        return self.img_tensor, self.lab_tensor

    def on_epoch_end(self):
        np.random.shuffle(self.image_names)
        # print('\033[0;37;40m on_epoch_end self.image_names = \n{}\033[0m'.format(self.image_names))


    def read_img(self, path):
        # print('path : ', path)
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.cast(img, tf.float32)
        img = img / 127.5 - 1
        return img

    def read_lab(self, path, is_edge_lab=False):
        # print('path : ', path)

        lab = tf.io.read_file(path)
        lab = tf.image.decode_png(lab, channels=1)
        if not is_edge_lab:
            lab /= 255
        return lab

