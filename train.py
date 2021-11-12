import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import backend as K
from edge_model import myModel
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy

from uility.load_data import load_train_func, load_val_func, Load_data
from uility.my_loss import Edge_Loss_V1, Build_Loss, Edge_Loss_V2, Edge_Loss_V3
import cv2 as cv
import matplotlib.pyplot as plt
from config import *

from uility.my_metrics import Recall, precision
from pred import my_pred


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':
    train_dataset = Load_data('train')
    train_step = len(train_dataset)
    val_dataset = Load_data('val')
    val_step = len(val_dataset)
    print('steps', train_step, val_step)


    # edge_loss = Edge_Loss_V1()
    edge_loss = Edge_Loss_V2()
    build_loss = Build_Loss()
    recall = Recall()

    model = myModel(backbone='resnet')
    model.summary()

    # model.compile(optimizer='Adam', loss=build_loss,
    #               metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss={'build_pred':build_loss,
                        "edge_pred1":edge_loss,
                        # "edge_pred2":edge_loss,
                        # "edge_pred3":edge_loss
                        },
                  loss_weights={'build_pred':1,
                        "edge_pred1":1,
                        # "edge_pred2":0.3,
                        # "edge_pred3":0.2
                                },
                  metrics={'build_pred': [precision],
                           # 'edge_pred1': ['accuracy'],
                           # 'edge_pred2': ['accuracy'],
                           # 'edge_pred3': ['accuracy'],
                           },

                  )

    model.fit(train_dataset,
              epochs=10,
              initial_epoch=0,
              steps_per_epoch=train_step,
              validation_data=val_dataset,
              validation_steps=5,
              )

    # pred
    my_pred(model)