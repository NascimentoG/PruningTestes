from sklearn.metrics import accuracy_score, top_k_accuracy_score
import numpy as np
from sklearn.utils import gen_batches
import keras
from keras.utils.generic_utils import CustomObjectScope
from tensorflow.data import Dataset
import h5py
import sys
import argparse
sys.path.insert(0, '../utils')
sys.path.insert(0, '../architectures')
import custom_functions as func
import rebuild_filters as rf


def count_res_blocks(model):
    #Returns the last Add of each block
    res_blocks = {}

    for layer in model.layers:
        if isinstance(layer, keras.layers.Add):
            dim = layer.output_shape[1]#1 and 2 are the spatial dimensions
            res_blocks[dim] = res_blocks.get(dim, 0) + 1

    return list(res_blocks.values())

def statistics(model):
    n_params = model.count_params()
    n_filters = func.count_filters(model)
    filter_layer = func.count_filters_layer(model)
    flops, _ = func.compute_flops(model)
    blocks = count_res_blocks(model)

    memory = func.memory_usage(1, model)
    print('Blocks {} Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] '
          'Memory [{:.6f}]'.format(blocks, n_params, n_filters, flops, memory), flush=True)

if __name__ == '__main__':
    np.random.seed(12227)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str,
                        default='/home/vm02/Documents/Gustavo/PruningMultipleStructuresImageNet224/ResNet50')
    parser.add_argument('--weights', type=str,
                        default='/home/vm02/Documents/Gustavo/PruningMultipleStructuresImageNet224/ResNet50')

    args = parser.parse_args()
    architecture_name = args.architecture
    weights = args.weights

    rf.architecture_name = architecture_name
    
    if architecture_name.__contains__('VGG16'):
        from tensorflow.keras.applications.vgg16 import preprocess_input

    if architecture_name.__contains__('MobileNet'):
        from tensorflow.keras.applications.mobilenet import preprocess_input

    if architecture_name.__contains__('ResNet'):
        from tensorflow.keras.applications.resnet50 import preprocess_input

    if architecture_name.__contains__('NASNet'):
        from tensorflow.keras.applications.inception_v3 import preprocess_input

    tmp = h5py.File('/home/vm02/Desktop/imageNet_images.h5', 'r')

    model = func.load_model(architecture_name)
    statistics(model)
    model.summary()
    # allowed_layers_filters = rf.layer_to_prune_filters(model)
    
    
    # X_test, y_test = tmp['X_test'], tmp['y_test']
    # y_pred = np.zeros((X_test.shape[0], y_test.shape[1]))

    # for batch in gen_batches(X_test.shape[0], 256):  # 1024 stands for the number of samples in primary memory
    #     samples = preprocess_input(X_test[batch].astype(float))

    #     # with tf.device("CPU"):
    #     X_tmp = Dataset.from_tensor_slices((samples)).batch(256)

    #     y_pred[batch] = model.predict(X_tmp, batch_size=256, verbose=0)

    # top1 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=1)
    # top5 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=5)
    # top10 = top_k_accuracy_score(np.argmax(y_test, axis=1), y_pred, k=10)
    # print('Top1 [{:.4f}] Top5 [{:.4f}] Top10 [{:.4f}]'.format(top1, top5, top10), flush=True)