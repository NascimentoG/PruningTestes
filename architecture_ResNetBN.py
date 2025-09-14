import numpy as np
import h5py
import sys

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers, models
from tensorflow.keras.layers import *
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.utils import gen_batches
from sklearn.metrics import accuracy_score

# ===========================
# Definições de blocos
# ===========================

def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def resnet(input_shape=(224, 224, 3), blocks=None, num_classes=1000):
    stacks = len(blocks)
    num_filters = 64
    bn_axis = 3

    img_input = Input(input_shape)
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = Conv2D(num_filters, 7, strides=2, use_bias=True, name='conv1_conv')(x)

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    for stage in range(stacks):
        num_res_blocks = blocks[stage]
        name = 'conv' + str(stage + 2)
        stride = 1 if stage == 0 else 2

        x = block1(x, filters=num_filters, stride=stride, name=name + '_block1')
        for res_block in range(2, num_res_blocks + 1):
            x = block1(x, num_filters, conv_shortcut=False, name=name + '_block' + str(res_block))

        num_filters *= 2

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='probs')(x)
    inputs = img_input

    model = models.Model(inputs, x, name='ResNetBN')
    return model


# ===========================
# Execução principal
# ===========================
if __name__ == '__main__':
    np.random.seed(12227)

    input_shape = (224, 224, 3)
    num_classes = 1000
    blocks = [3, 4, 6, 3]

    model = resnet(input_shape=input_shape, blocks=blocks, num_classes=num_classes)

    # Ajuste de caminho dos pesos (se já baixados)
    # model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    # Dataset (ajuste para seu ambiente)
    tmp = h5py.File('E:/datasets/ImageNet/processed_data/imageNet_images.h5', 'r')
    X_test, y_test = tmp['X_test'], tmp['y_test']

    y_pred = np.zeros((X_test.shape[0], y_test.shape[1]))
    for batch in gen_batches(X_test.shape[0], 1024):
        samples = preprocess_input(X_test[batch].astype(float))
        y_pred[batch] = model.predict(samples, batch_size=256)

    # Top-1 e Top-k accuracies
    top1_error = 1 - accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    def top_k_accuracy(y_true, y_pred, k=5):
        return np.mean(tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=k))

    top5_error = 1 - top_k_accuracy(y_test, y_pred, 5)
    top10_error = 1 - top_k_accuracy(y_test, y_pred, 10)

    print(f'Top1 [{top1_error:.4f}] Top5 [{top5_error:.4f}] Top10 [{top10_error:.4f}]')