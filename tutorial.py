import os
import sys
import argparse

import numpy as np
import tensorflow as tf
import architecture_ResNetBN as arch
import custom_callbacks
import custom_functions as func
import rebuild_layers as rl
import rebuild_filters as rf
import criteria_filter as cf
import criteria_layer as cl

from datetime import datetime

from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.utils import gen_batches

import keras

from tensorflow.data import Dataset
import h5py
from tensorflow.keras.applications.resnet50 import preprocess_input

from typing import Tuple, List, Dict


n_samples_per_class = 5
NUM_CLASSES = 1000   
INPUT_SHAPE = (224, 224, 3)
_RNG = np.random.default_rng(42)  

X_list, y_list = [], []

for c in range(NUM_CLASSES):
    X_c = _RNG.normal(size=(n_samples_per_class, *INPUT_SHAPE)).astype("float32")
    y_c = tf.one_hot([c] * n_samples_per_class, depth=NUM_CLASSES).numpy()

    X_list.append(X_c)
    y_list.append(y_c)

X_train_sampled = np.vstack(X_list)
y_train_sampled = np.vstack(y_list)

indices = np.arange(len(y_train_sampled))
np.random.shuffle(indices)
X_train_sampled = X_train_sampled[indices]
y_train_sampled = y_train_sampled[indices]

X_test_list, y_test_list = [], []

for c in range(NUM_CLASSES):
    X_c = _RNG.normal(size=(1, *INPUT_SHAPE)).astype("float32")  # 1 amostra por classe
    y_c = tf.one_hot([c], depth=NUM_CLASSES).numpy()

    X_test_list.append(X_c)
    y_test_list.append(y_c)

X_test_sampled = np.vstack(X_test_list)
y_test_sampled = np.vstack(y_test_list)

X_train_sampled = X_train_sampled
y_train_sampled = y_train_sampled
X_test = X_test_sampled
y_test = y_test_sampled

architecture_name = 'ResNet50'

rf.architecture_name = architecture_name
rl.architecture_name = architecture_name

def pruneByLayer(model, criteria, p_layer):
    allowed_layers = rl.blocks_to_prune(model)
    layer_method = cl.criteria(criteria)
    scores = layer_method.scores(model, X_train_sampled, y_train_sampled, allowed_layers)    
    
    return rl.rebuild_network(model, scores, p_layer)

def pruneByFilter(model, criteria, p_filter):
    allowed_layers_filters = rf.layer_to_prune_filters(model)
    numberToFilterToRemove = int(func.count_filters(model)*p_filter)
    filter_method = cf.criteria(criteria)
    scores = filter_method.scores(model, X_train_sampled, y_train_sampled, allowed_layers_filters)    
    
    return  rf.rebuild_network(model, scores, p_filter, numberToFilterToRemove)

def prediction(model, X_test, y_test):
    y_pred = np.zeros((X_test.shape[0], y_test.shape[1]))

    for batch in gen_batches(X_test.shape[0], 256):  # 256 stands for the number of samples in primary memory
        samples = preprocess_input(X_test[batch].astype(float))

        # with tf.device("CPU"):
        X_tmp = Dataset.from_tensor_slices((samples)).batch(256)

        y_pred[batch] = model.predict(X_tmp, batch_size=256, verbose=0)

    top1 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=1)
    top5 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=5)
    top10 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=10)
    #print('Top1 [{:.4f}] Top5 [{:.4f}] Top10 [{:.4f}]'.format(top1, top5, top10), flush=True)
    
    return top1
          
def statistics(model):
    acc = prediction(model, X_test, y_test)
    
    n_params = model.count_params()
    n_filters = func.count_filters(model)
    filter_layer = func.count_filters_layer(model)
    flops, _ = func.compute_flops(model)
    blocks = rl.count_res_blocks(model)

    memory = func.memory_usage(1, model)

    print('Accuracy [{}] Blocks {} Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] '
        'Memory [{:.6f}]'.format(acc, blocks, n_params, n_filters, flops, memory), flush=True)

