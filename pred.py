from config import *
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt


def my_pred(model):
    img = cv.imread(PRED_PATH)
    img = tf.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32)
    img = img / 127.5 - 1
    res = model(img)
    print(res)
    # res = tf.where(res > 0.5, 1, 0)
    # print(res)
    threshold = THRESHOLD
    build_pred = tf.where(res[0] > threshold, 1, 0)
    edge_pred1 = tf.where(res[1] > threshold, 1, 0)
    # edge_pred2 = tf.where(res[2] > threshold, 1, 0)
    # edge_pred3 = tf.where(res[3] > threshold, 1, 0)

    plt.figure(figsize=(8, 8), dpi=300, facecolor='#FFFFFF', edgecolor='#0000FF')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 修改字体
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(1)

    ax1 = plt.subplot(221)
    plt.xlabel('图a build_pred', fontsize=12)
    plt.imshow(tf.squeeze(build_pred))

    ax2 = plt.subplot(222)
    plt.xlabel('图b edge_pred1', fontsize=12)
    plt.imshow(tf.squeeze(edge_pred1))

    # ax3 = plt.subplot(223)
    # plt.xlabel('图c edge_pred2', fontsize=12)
    # plt.imshow(tf.squeeze(edge_pred2))
    #
    # ax4 = plt.subplot(224)
    # plt.xlabel('图d edge_pred3', fontsize=12)
    # plt.imshow(tf.squeeze(edge_pred3))
    plt.show()