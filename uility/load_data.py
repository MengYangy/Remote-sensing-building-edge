import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from config import *
import numpy as np
import cv2 as cv
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
    '''
    功能：边缘损失模型数据生成器，包含img,lab,edge_lab
    '''
    def __init__(self, type='train'):
        self.batch_size = Batch_Size
        self.img_path = IMAGE_PATH if type == 'train' else VAL_IMAGE_PATH
        self.lab_path = LABEL_PATH if type == 'train' else VAL_LABEL_PATH
        self.edge_path = EDGE_LABEL_PATH if type == 'train' else VAL_EDGE_LABEL_PATH


        self.img_names = os.listdir(self.img_path)
        self.img_nums = len(self.img_names)
        self.img_names = np.array(self.img_names)
        self.random_list = np.arange(self.img_nums)
        np.random.shuffle(self.random_list)

    def __getitem__(self, item):
        img_name_list = self.img_names[item*self.batch_size:(item+1)*self.batch_size]
        img_tensor = self.read_img(img_name_list)
        lab_tensor = self.read_lab(img_name_list)
        return img_tensor, lab_tensor
        # return img_name_list

    def __len__(self):
        return math.floor(self.img_nums / self.batch_size)

    def read_img(self, img_name_list):
        img_tensor = np.zeros((self.batch_size, IMG_H, IMG_W, 3))
        for i in range(len(img_name_list)):
            img = cv.imread(os.path.join(self.img_path, img_name_list[i]))
            img_tensor[i, :, :, :] = img
        img_tensor = img_tensor / 127.5 - 1.0
        return img_tensor

    def read_lab(self, img_name_list):
        lab_tensor = np.zeros((self.batch_size, IMG_H, IMG_W, 2))
        for i in range(len(img_name_list)):
            lab = cv.imread(os.path.join(self.lab_path, img_name_list[i]))
            edge = cv.imread(os.path.join(self.edge_path, img_name_list[i]))
            lab = cv.cvtColor(lab, cv.COLOR_BGR2GRAY)
            edge = cv.cvtColor(edge, cv.COLOR_BGR2GRAY)
            lab = np.reshape(lab, (IMG_H, IMG_W, 1))
            edge = np.reshape(edge, (IMG_H, IMG_W, 1))
            lab_tensor[i, :, :, 0:1] = lab
            lab_tensor[i, :, :, 1:2] = edge
        lab_tensor[:, :, :, 0:1] = lab_tensor[:, :, :, 0:1] / 255
        return lab_tensor

if __name__ == '__main__':
    fun = Load_data(type='val')
    for img, lab in fun:
        print(img.shape, lab.shape)
    labs = fun.read_lab(['val_123.png'])
    print(labs[0].shape)

    VAL_IMAGE_PATH = 'D:/dataset/newedge/val/image'
    VAL_LABEL_PATH = 'D:/dataset/newedge/val/label'
    VAL_EDGE_LABEL_PATH = 'D:/dataset/newedge/val/edgelabel'
    lab = cv.imread(os.path.join(VAL_LABEL_PATH, 'val_123.png'))
    edge = cv.imread(os.path.join(VAL_EDGE_LABEL_PATH, 'val_123.png'))
    lab = cv.cvtColor(lab, cv.COLOR_BGR2GRAY)
    edge = cv.cvtColor(edge, cv.COLOR_BGR2GRAY)
    lab = np.reshape(lab, (IMG_H, IMG_W, 1))
    edge = np.reshape(edge, (IMG_H, IMG_W, 1))
    print(edge, edge.shape)
    print(labs[0, :, :, 1], labs[0, :, :, 1].shape)
    print(np.unique(labs[0, :, :, 1:2] - edge))
