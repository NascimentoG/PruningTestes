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

class NeuralNetwork():
    def __init__(self):
        self.model = None
        self.modelsFilter = []
        self.modelsLayer = []
        self.architecture_name = None
        self.criterion_layer = None
        self.criterion_filter = None
        self.p_filter = None
        self.p_layer = None
        self.acc = None
        
        self.wasPfilterZero = False
        self.directory = 'saved_models/saved_model_' + datetime.now().strftime("%Y_%m_%d")

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            print(f'A pasta "{self.directory}" foi criada.')
        else:
            print(f'A pasta "{self.directory}" já existe.')
        
    def statistics(self, model, i, acc):
        n_params = model.count_params()
        n_filters = func.count_filters(model)
        filter_layer = func.count_filters_layer(model)
        flops, _ = func.compute_flops(model)
        blocks = rl.count_res_blocks(model)

        memory = func.memory_usage(1, model)
        print('Iteration [{}] Accuracy [{}] Blocks {} Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] '
            'Memory [{:.6f}]'.format(i, acc, blocks, n_params, n_filters, flops, memory), flush=True)    
    
    def prediction(self, model, X_test, y_test):
        y_pred = np.zeros((X_test.shape[0], y_test.shape[1]))

        for batch in gen_batches(X_test.shape[0], 256):  # 256 stands for the number of samples in primary memory
            samples = preprocess_input(X_test[batch].astype(float))

            # with tf.device("CPU"):
            X_tmp = Dataset.from_tensor_slices((samples)).batch(256)

            y_pred[batch] = model.predict(X_tmp, batch_size=256, verbose=0)

        top1 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=1)
        top5 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=5)
        top10 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=10)
        print('Top1 [{:.4f}] Top5 [{:.4f}] Top10 [{:.4f}]'.format(top1, top5, top10), flush=True)
        
        return top1

    def finetuning(self, model, X_train, y_train, X_test, y_test):
        lr = 0.001
        schedule = [(2, 1e-4), (4, 1e-5)]

        #It checks if the code saves the model correctly
        # func.save_model('Criterion[{}]_Filters{}_P[{}]_Epoch{}'.format(criterion_filter, func.count_filters(model), p_filter, 0), model)

        lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
        callbacks = [lr_scheduler]

        sgd = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        for ep in range(0, 5):
            for batch in gen_batches(X_train.shape[0], 1024):
                samples = func.data_augmentation(X_train[batch].astype(float), padding=28)
                samples = preprocess_input(samples)

                X_tmp = Dataset.from_tensor_slices((samples, y_train[batch])).shuffle(4 * 64).batch(64)

                model.fit(X_tmp,
                        callbacks=callbacks, verbose=2,
                        epochs=ep, initial_epoch=ep - 1,
                        batch_size=64)
            if ep % 3:
                acc = self.prediction(model, X_test, y_test)
                print(f"Accuracy {acc}")
            #if ep in [3, 5]:
            # func.save_model('Criterion[{}]_Filters[{}]_P[{}]_Epoch{}'.format(criterion_filter, func.count_filters(model), p_filter, ep), model)

        return model
    
    def finetuningCKA(self, model, epochs):
        lr = 0.001
        schedule = [(2, 1e-4), (4, 1e-5)]

        lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
        callbacks = [lr_scheduler]
        
        sgd = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        for ep in range(0, epochs):
            for batch in gen_batches(self.X_train.shape[0], 1024):
                samples = func.data_augmentation(self.X_train[batch].astype(float), padding=28)
                samples = preprocess_input(samples)

                X_tmp = Dataset.from_tensor_slices((samples, self.y_train[batch])).shuffle(4 * 64).batch(64)

                model.fit(X_tmp,
                        callbacks=callbacks, verbose=2,
                        epochs=ep, initial_epoch=ep - 1,
                        batch_size=64)
            if ep % 3:
                acc = self.prediction(model, self.X_test, self.y_test)
                print(f"Accuracy {acc}")
                
            #if ep in [3, 5]:
            # func.save_model('Criterion[{}]_Filters[{}]_P[{}]_Epoch{}'.format(criterion_filter, func.count_filters(model), p_filter, ep), model)

        return model
    
    def feature_space_linear_cka(self, features_x, features_y, debiased=False):
        features_x = features_x - np.mean(features_x, 0, keepdims=True)
        features_y = features_y - np.mean(features_y, 0, keepdims=True)

        dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
        normalization_x = np.linalg.norm(features_x.T.dot(features_x))
        normalization_y = np.linalg.norm(features_y.T.dot(features_y))

        if debiased:
            n = features_x.shape[0]
            # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
            sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
            sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
            squared_norm_x = np.sum(sum_squared_rows_x)
            squared_norm_y = np.sum(sum_squared_rows_y)

            dot_product_similarity = self._debiased_dot_product_similarity_helper(
                dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
                squared_norm_x, squared_norm_y, n)
            normalization_x = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
                squared_norm_x, squared_norm_x, n))
            normalization_y = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
                squared_norm_y, squared_norm_y, n))

        return dot_product_similarity / (normalization_x * normalization_y)
    
    def feature_extraction(self, model, X):
        features = np.zeros((X.shape[0], model.output_shape[-1]))

        for batch in gen_batches(X.shape[0], 128):  # 1024 stands for the number of samples in primary memory
            samples = preprocess_input(X[batch].astype(float))
            features[batch] = model.predict(samples, batch_size=128, verbose=0)

        return features
    
    def CKA(self, unpruned, pruned):
        bestScore = ["",0.0]
        bestModel = None

        F = keras.Model(unpruned.input, unpruned.get_layer(index=-2).output)
        # featuresUnpruned = F.predict(neuralNetwork.X_test_sampled, verbose=0)
        featuresUnpruned = self.feature_extraction(F, self.X_test_sampled)
        
        for criteria, model in pruned:
            print("Criteria: " + str(criteria))
            model = self.finetuningCKA(model, 1)
            
            F = keras.Model(model.input, model.get_layer(index=-2).output)
            # featuresPruned = F.predict(neuralNetwork.X_test_sampled, verbose=0)
            featuresPruned = self.feature_extraction(F, self.X_test_sampled)
            
            score = self.feature_space_linear_cka(featuresUnpruned, featuresPruned)
            print(f"Score {score}")
            if score > bestScore[1]:
                bestScore[0] = criteria
                bestScore[1] = score
                bestModel = model
                
        return bestScore[0], bestScore[1], bestModel
         
    def selectPrunedNetwork(self, selectedCriteriaLayer, scoreMetricLayer, bestModelLayer, selectedCriteriaFilter, scoreMetricFilter, bestModelFilter):
        if self.wasPfilterZero:
            print("O P_Filter era 0 porque nao é possível remover mais camadas")
        
        if (scoreMetricLayer >= scoreMetricFilter) and not self.wasPfilterZero:
            self.model = bestModelLayer
            self.usedCriteria = selectedCriteriaLayer
            self.prunedType = "layer"
        else:
            self.model = bestModelFilter
            self.usedCriteria = selectedCriteriaFilter
            self.prunedType = "filter"
            
    def clearVariables(self):
        self.modelsFilter = []
        self.modelsLayer = []
        self.wasPfilterZero = False
        
    def saveCKALog(self, iteration, scoreLayer,scoreFilter,selectedMode):
        with open(self.directory + ".txt", "a") as arquivo:
            arquivo.write(f"{iteration} {scoreLayer} {scoreFilter} {selectedMode}\n")
    
    
if __name__ == '__main__':
    neuralNetwork = NeuralNetwork()
        
    np.random.seed(12227)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='ResNet50')
    parser.add_argument('--criterion_layer', type=list, default=['klDivergence'])
    parser.add_argument('--criterion_filter', type=list, default=['klDivergence'])
    parser.add_argument('--p_layer', type=float, default=1)
    parser.add_argument('--p_filter', type=float, default=0.50)
    debug = False

    args = parser.parse_args()
    neuralNetwork.architecture_name = args.architecture
    neuralNetwork.p_layer = args.p_layer
    neuralNetwork.p_filter = args.p_filter
    neuralNetwork.criterion_layer = args.criterion_layer
    neuralNetwork.criterion_filter = args.criterion_filter
    
    # scores = []#Put the precomputed scores here
    cl.preprocess_input = preprocess_input

    rf.architecture_name = neuralNetwork.architecture_name
    rl.architecture_name = neuralNetwork.architecture_name

    print(args, flush=False)
    print('Architecture [{}] p_filter[{}] p_layer[{}]'.format(neuralNetwork.architecture_name, neuralNetwork.p_filter, neuralNetwork.p_layer), flush=True)

    if neuralNetwork.architecture_name == 'ResNet50':
        blocks = [3, 4, 6, 3]
        neuralNetwork.model = arch.resnet(input_shape=(224, 224, 3), blocks=blocks)

    if neuralNetwork.architecture_name == 'ResNet101':
        blocks = [3, 4, 23, 3]
        neuralNetwork.model = arch.resnet(input_shape=(224, 224, 3), blocks=blocks)

    if neuralNetwork.architecture_name == 'ResNet152':
        blocks = [3, 8, 36, 3]
        neuralNetwork.model = arch.resnet(input_shape=(224, 224, 3), blocks=blocks)

    if debug == False:
        neuralNetwork.model = func.load_model('{}'.format(neuralNetwork.architecture_name),
                             '{}'.format(neuralNetwork.architecture_name))
        tmp = h5py.File('/root/project_ws/media/gustavo/TOSHIBA EXT/ImageNet/imageNet_images.h5', 'r')
        
        neuralNetwork.X_train, neuralNetwork.y_train = tmp['X_train'], tmp['y_train']
        neuralNetwork.X_test, neuralNetwork.y_test = tmp['X_test'], tmp['y_test']
        
        n_samples = 10
        
        y_ = np.argmax(neuralNetwork.y_train, axis=1)
        sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                        np.unique(y_)]
        sub_sampling = np.array(sub_sampling).reshape(-1)
        sub_sampling = sorted(sub_sampling)
        neuralNetwork.X_train_sampled = neuralNetwork.X_train[sub_sampling]
        neuralNetwork.y_train_sampled = neuralNetwork.y_train[sub_sampling]
        # X_train_sampled = preprocess_input(X_train_sampled.astype(float))
        
        yt_ = np.argmax(neuralNetwork.y_test, axis=1)
        sub_sampling_test = [np.random.choice(np.where(yt_ == value)[0], n_samples, replace=False) for value in
                        np.unique(yt_)]
        sub_sampling_test = np.array(sub_sampling_test).reshape(-1)
        sub_sampling_test = sorted(sub_sampling_test)
        neuralNetwork.X_test_sampled = neuralNetwork.X_test[sub_sampling_test]
        neuralNetwork.y_test_sampled = neuralNetwork.y_test[sub_sampling_test]
        
        n_samples = None

    else:
        n_samples_per_class = 5
        n_classes = 1000
        resolution = 224

        # Generate at least one sample per class for training
        neuralNetwork.X_train_sampled = np.random.rand(n_samples_per_class * n_classes, resolution, resolution, 3)

        # Generate labels for training
        neuralNetwork.y_train_sampled = np.zeros((n_samples_per_class * n_classes, n_classes))
        neuralNetwork.y_train_sampled[np.arange(n_samples_per_class * n_classes), np.repeat(np.arange(n_classes), n_samples_per_class)] = 1

        # Shuffle training data
        indices = np.arange(n_samples_per_class * n_classes)
        np.random.shuffle(indices)
        neuralNetwork.X_train_sampled = neuralNetwork.X_train_sampled[indices]
        neuralNetwork.y_train_sampled = neuralNetwork.y_train_sampled[indices]

        # Generate random samples for testing
        neuralNetwork.X_test_sampled = np.random.rand(n_samples_per_class, resolution, resolution, 3)
        neuralNetwork.y_test_sampled = np.eye(n_classes)[np.random.randint(0, n_classes, n_samples_per_class)]

        #neuralNetwork.model = arch.resnet(input_shape=(resolution, resolution, 3), blocks=blocks) #Scale-wise
        neuralNetwork.model = func.load_model('{}'.format(neuralNetwork.architecture_name),
                             '{}'.format(neuralNetwork.architecture_name))
        
    for i in range(0, 40):
        
        if rl.count_res_blocks(neuralNetwork.model) == [2, 2, 2, 2]:
            print("Não é possível remover mais layers")
            break
        
        elif not rf.isFiltersAvailable:
            print("Só é possível remover layers")
            print("Pruning by layer")
            allowed_layers = rl.blocks_to_prune(neuralNetwork.model)
            for criteria in neuralNetwork.criterion_layer:
                print(f"{criteria}")
                layer_method = cl.criteria(criteria)
                scores = layer_method.scores(neuralNetwork.model, neuralNetwork.X_train_sampled, neuralNetwork.y_train_sampled, allowed_layers)    
                neuralNetwork.modelsLayer.append([criteria, rl.rebuild_network(neuralNetwork.model, scores, neuralNetwork.p_layer)])

            selectedCriteriaLayer, scoreMetricLayer, bestModelLayer = neuralNetwork.CKA(neuralNetwork.model, neuralNetwork.modelsLayer)
            neuralNetwork.usedCriteria = selectedCriteriaLayer
            neuralNetwork.prunedType = 'layer'
            neuralNetwork.model = bestModelLayer
        
        else:
            print("Possível remover layers e filtros")
            neuralNetwork.acc = neuralNetwork.prediction(neuralNetwork.model, neuralNetwork.X_test, neuralNetwork.y_test)
            neuralNetwork.statistics(neuralNetwork.model, i, neuralNetwork.acc)
            print("Pruning by layer")
            allowed_layers = rl.blocks_to_prune(neuralNetwork.model)
            for criteria in neuralNetwork.criterion_layer:
                print(f"{criteria}")
                layer_method = cl.criteria(criteria)
                scores = layer_method.scores(neuralNetwork.model, neuralNetwork.X_train_sampled, neuralNetwork.y_train_sampled, allowed_layers)    
                neuralNetwork.modelsLayer.append([criteria, rl.rebuild_network(neuralNetwork.model, scores, neuralNetwork.p_layer)])

            neuralNetwork.quantityRemoved = (func.count_filters(neuralNetwork.model) - func.count_filters(neuralNetwork.modelsLayer[0][1]))
            neuralNetwork.p_filter = (neuralNetwork.quantityRemoved)/(func.count_filters(neuralNetwork.model))
            
            if neuralNetwork.p_filter == 0:
                neuralNetwork.wasPfilterZero = True
                neuralNetwork.p_filter = 0.5
                neuralNetwork.quantityRemoved = int((func.count_filters(neuralNetwork.model))*neuralNetwork.p_filter)
                print(f"Já podamos todas as camadas, definindo o p_filter para {neuralNetwork.p_filter}, vamos remover {neuralNetwork.quantityRemoved}")

            print(f"No modelo original, temos {func.count_filters(neuralNetwork.model)}, no podado por layer temos {func.count_filters(neuralNetwork.modelsLayer[0][1])}, logo vamos setar para remover {neuralNetwork.p_filter} %")
            print("Pruning by filter")
            allowed_layers_filters = rf.layer_to_prune_filters(neuralNetwork.model)
        
            for criteria in neuralNetwork.criterion_filter:
                filter_method = cf.criteria(criteria)
                scores = filter_method.scores(neuralNetwork.model, neuralNetwork.X_train_sampled, neuralNetwork.y_train_sampled, allowed_layers_filters)    
                neuralNetwork.modelsFilter.append([criteria, rf.rebuild_network(neuralNetwork.model, scores, neuralNetwork.p_filter, neuralNetwork.quantityRemoved, neuralNetwork.wasPfilterZero)])
            
            print(f"No modelo original, temos {func.count_filters(neuralNetwork.model)}, no podado por filtros temos {func.count_filters(neuralNetwork.modelsFilter[0][1])}")
        
            selectedCriteriaLayer, scoreMetricLayer, bestModelLayer = neuralNetwork.CKA(neuralNetwork.model, neuralNetwork.modelsLayer)
            selectedCriteriaFilter, scoreMetricFilter, bestModelFilter = neuralNetwork.CKA(neuralNetwork.model, neuralNetwork.modelsFilter)
            
            neuralNetwork.selectPrunedNetwork(selectedCriteriaLayer, scoreMetricLayer, bestModelLayer, selectedCriteriaFilter, scoreMetricFilter, bestModelFilter)
            
            neuralNetwork.saveCKALog(i,scoreMetricLayer, scoreMetricFilter, neuralNetwork.prunedType)
            print(f"Select pruned type {neuralNetwork.prunedType}, using {neuralNetwork.usedCriteria}")
            
        neuralNetwork.model = neuralNetwork.finetuning(neuralNetwork.model, neuralNetwork.X_train, neuralNetwork.y_train, neuralNetwork.X_test, neuralNetwork.y_test)#or neuralNetwork.model = finetuning(pruned_model_layer,...)

        neuralNetwork.acc = neuralNetwork.prediction(neuralNetwork.model, neuralNetwork.X_test, neuralNetwork.y_test)

        neuralNetwork.statistics(neuralNetwork.model, i, neuralNetwork.acc)
        # meanLattency = func.meanLattency(neuralNetwork.model,neuralNetwork.X_test)
        
        # print(f"A latencia média é de {meanLattency}")
        func.save_model(neuralNetwork.directory + '/{}_{}_{}_{}_iterations[{}]'.format(i,neuralNetwork.architecture_name, neuralNetwork.usedCriteria, neuralNetwork.prunedType, i), neuralNetwork.model)

        neuralNetwork.clearVariables()
