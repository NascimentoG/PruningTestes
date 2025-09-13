import csv
import os
import sys
import argparse

import numpy as np

from datetime import datetime

from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.utils import gen_batches
import sys
import keras
import tensorflow as tf
from tensorflow.data import Dataset
import h5py
from tensorflow.keras.applications.resnet50 import preprocess_input
import argparse

import rebuild_layers as rl
import rebuild_filters as rf

from pruning_criteria import criteria_filter as cf
from pruning_criteria import criteria_layer as cl

sys.path.insert(0, '../utils')

import custom_functions as func
import custom_callbacks
import architecture_ResNetBN as arch

import argparse
gpu_dict = {
    'RTX 4090': 450,
    'RTX 4080': 320,
    'RTX 4070 TI': 285,
    'RTX 4070': 200,
    'RTX 4060 TI': 160,
    'RTX 4060': 115,
    'RTX 3090 TI': 450,
    'RTX 3090': 350,
    'RTX 3080 TI': 350,
    'RTX 3080 12GB': 350,
    'RTX 3080 10GB': 320,
    'RTX 3070 TI': 290,
    'RTX 3070': 220,
    'RTX 3060 TI': 200,
    'RTX 3060': 170,
    'RTX 3050': 130,
    'RTX 2080 TI': 250,
    'RTX 2080 Super': 250,
    'RTX 2080': 215,
    'RTX 2070 Super': 215,
    'RTX 2070': 175,
    'RTX 2060 Super': 175,
    'RTX 2060': 160,
    'GTX 1660 TI': 120,
    'GTX 1660 Super': 125,
    'GTX 1660': 120,
    'GTX 1650 Super': 100,
    'GTX 1650': 75,
    'GTX 1080 TI': 250,
    'GTX 1080': 180,
    'GTX 1070 TI': 180,
    'GTX 1070': 150,
    'GTX 1060 6GB': 120,
    'GTX 1060 3GB': 120,
    'GTX 1050 TI': 75,
    'GTX 1050': 75
}

def saveOnCSV(file, data):
    with open(file, mode='a', newline='') as arquivo_csv:
        escritor_csv = csv.writer(arquivo_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        escritor_csv.writerow(data)

def lattency(model, X_test):
    pred = model.predict(X_test, verbose=0)
    inferenceTime = 0
    for i in range(3): 
        init =  time.time()
        pred = model.predict(X_test, verbose=1)
        inferenceTime += time.time() - init
    print(inferenceTime/(3*len(X_test)))
    return inferenceTime/(3*len(X_test))

def finetuning(epochs, model, X_train, y_train, X_test, y_test):
        lr = 0.01
        schedule = [(100, lr / 10), (150, lr / 100)]
        lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
        callbacks = [lr_scheduler]

        sgd = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test, verbose=0), axis=1))
        print('Accuracy before fine-tuning  [{:.4f}]'.format(acc), flush=True)

        inferenceTime = 0
        
        for ep in range(0, epochs):
            y_tmp = np.concatenate((y_train, y_train, y_train))
            X_tmp = np.concatenate(
                (func.data_augmentation(X_train),
                func.data_augmentation(X_train),
                func.data_augmentation(X_train)))

            with tf.device("CPU"):
                X_tmp = Dataset.from_tensor_slices((X_tmp, y_tmp)).shuffle(4 * 128).batch(128)

            if ep == 0:
                model.fit(X_tmp, batch_size=128,
                    callbacks=callbacks, verbose=2,
                    epochs=ep, initial_epoch=ep - 1)
            else:
                init =  time.time()
                model.fit(X_tmp, batch_size=128,
                    callbacks=callbacks, verbose=2,
                    epochs=ep, initial_epoch=ep - 1)
                inferenceTime += time.time() - init


        return inferenceTime

def greenAI(ft_time, epochs, gpu, numero_gpus, custo_kwh, gCO2_por_KWH):
    horas = ft_time*epochs/3600
    
    watts_gpu = gpu_dict[gpu]
    kwh = horas * watts_gpu * numero_gpus / 1000

    custo = kwh * custo_kwh
    gCO2 = kwh * gCO2_por_KWH
    
    return kwh, custo, gCO2
 
def statistics(architecture_name, modelName, model, i, acc, X_train, y_train, X_test, y_test, gpu, numero_gpus, custo_kwh, gCO2_por_KWH):
    n_params = model.count_params()
    n_filters = func.count_filters(model)
    filter_layer = func.count_filters_layer(model)
    flops, _ = func.compute_flops(model)
    blocks = rl.count_res_blocks(model)

    memory = func.memory_usage(1, model)
    #latency = lattency(model, X_test)
    #ft_time = finetuning(7, model, X_train, y_train, X_test, y_test)
    #kwh, custo, gCO2 = greenAI(ft_time, 200, gpu, numero_gpus, custo_kwh, gCO2_por_KWH)
    
    print('Accuracy [{}] Blocks {} Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] '
        'Memory [{:.6f}]'.format(acc, blocks, n_params, n_filters, flops, memory), flush=True)
    
    #print('kWh: ', round(kwh, 2), '| Custo: R$', round(custo, 2), ' | gCO2: ', round(gCO2, 2))

    #print('Meses de uma árvore para sequestrar o CO2: ', round(kwh * 0.17 / 2.53, 2), '\n'
    #                                                                                  'Equivalente de CO2 em quilometros rodados em um carro de passageiro comum: ',
    #      round(kwh * 0.89 / 2.53, 2), '\n'
    #                                   'Equivalente de CO2 em porcentagem de um voo paris-londres: ',
    #      round(kwh * 0.31 / 2.53, 2), '% ')
    
    saveOnCSV(architecture_name + ".csv", [modelName, acc, n_params, n_filters, flops, memory])#, ((ft_time*200)/3600), round(kwh, 2), round(custo, 2),
                                           #round(gCO2, 2), round(kwh * 0.17 / 2.53, 2), round(kwh * 0.89 / 2.53, 2), round(kwh * 0.31 / 2.53, 2)])

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture_name', type=str, default='ResNet50')
    parser.add_argument('--saved_models_dir', type=str, default='ResNet50_2')
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='RTX 3060')
    parser.add_argument('--custo_kwh', type=float, default=0.67)
    parser.add_argument('--gCO2', type=float, default=38.5)
    parser.add_argument('--n_gpus', type=int, default=1)
    
    args = parser.parse_args()
    
    savedModelsDir = "saved_models/" + args.saved_models_dir
    architecture_name = args.architecture_name
    
    gpu = args.gpu
    custo_kwh = args.custo_kwh
    gCO2_por_KWH = args.gCO2
    numero_gpus = args.n_gpus
    
    rf.architecture_name = architecture_name
    rl.architecture_name = architecture_name
    
    
    savedModels = []
    savedModelsWeights = []
    savedModelsFiles = []
    
    if not os.path.exists(savedModelsDir):
        print(f'Diretorio inexistente: {savedModelsDir}')
    else:
        savedModelsFiles = os.listdir(savedModelsDir)
        savedModels = sorted([model for model in savedModelsFiles if model[-5:] == '.json'])
        savedModelsWeights = sorted([model for model in savedModelsFiles if model[-3:] == '.h5'])
        
        print(savedModels)
        print(savedModelsWeights)
            
    tmp = h5py.File('/root/project_ws/media/gustavo/TOSHIBA EXT/ImageNet/imageNet_images.h5', 'r')
        
    X_train, y_train = tmp['X_train'], tmp['y_train']
    X_test, y_test = tmp['X_test'], tmp['y_test']
    
    n_samples = 10
    
    y_ = np.argmax(y_train, axis=1)
    sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                    np.unique(y_)]
    sub_sampling = np.array(sub_sampling).reshape(-1)
    sub_sampling = sorted(sub_sampling)
    X_train_sampled = X_train[sub_sampling]
    y_train_sampled = y_train[sub_sampling]
    # X_train_sampled = preprocess_input(X_train_sampled.astype(float))
    
    yt_ = np.argmax(y_test, axis=1)
    sub_sampling_test = [np.random.choice(np.where(yt_ == value)[0], n_samples, replace=False) for value in
                    np.unique(yt_)]
    sub_sampling_test = np.array(sub_sampling_test).reshape(-1)
    sub_sampling_test = sorted(sub_sampling_test)
    X_test_sampled = X_test[sub_sampling_test]
    y_test_sampled = y_test[sub_sampling_test]
    
    
    originalModel = func.load_model('{}'.format(architecture_name),
                            '{}'.format(architecture_name))
    
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(originalModel.predict(X_test, verbose=0), axis=1))
    statistics(architecture_name, architecture_name, originalModel, 'Unpruned', acc, X_train_sampled, y_train_sampled, X_test_sampled, y_test_sampled, gpu, numero_gpus, custo_kwh, gCO2_por_KWH)

    for model , weight in zip(savedModels, savedModelsWeights):
        currentModel = func.load_model('{}/{}'.format(savedModelsDir,model[:-5]),
                                '{}/{}'.format(savedModelsDir, weight[:-3]))
        
        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(currentModel.predict(X_test, verbose=0), axis=1))
        statistics(architecture_name, model[:-5], currentModel, 'Pruned', acc, X_train_sampled, y_train_sampled, X_test_sampled, y_test_sampled, gpu, numero_gpus, custo_kwh, gCO2_por_KWH)
    
