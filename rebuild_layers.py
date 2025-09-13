import numpy as np
from sklearn.utils import gen_batches
import gc
import sys
import copy

from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import Input
import architecture_ResNetBN as arch

def filters_layerResNet50(model, mask):
    output = []

    # Add the same weights until finding the first Add layer
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D):
            output.append((i, layer.output_shape[-1]))

        if isinstance(layer, Add):
            break

    add_model = blocks_to_prune(model)
    add_model = np.array(add_model)[mask == 1]
    add_model = list(add_model)

    for layer_idx in range(0, len(add_model)):
        idx_model = np.arange(add_model[layer_idx] - 9, add_model[layer_idx] + 1).tolist()
        for i in idx_model:
            layer = model.get_layer(index=i)
            if isinstance(layer, Conv2D):
                output.append((i, layer.output_shape[-1]))

    add_model = add_to_downsampling(model)
    for layer_idx in range(0, len(add_model)):
        idx_model = np.arange(add_model[layer_idx] - 11, add_model[layer_idx] + 1).tolist()
        for i in idx_model:
            layer = model.get_layer(index=i)
            if isinstance(layer, Conv2D):
                output.append((i, layer.output_shape[-1]))

    output.sort(key=lambda tup: tup[0])
    output = [item[1] for item in output]
    return output

def blocks_to_prune(model):
    allowed_layers = []
    all_add = []

    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Add):
            all_add.append(i)

    for i in range(1, len(all_add) - 1):
        input_shape = model.get_layer(index=all_add[i]).output_shape
        output_shape = model.get_layer(index=all_add[i - 1]).output_shape
        # These are the valid blocks we can remove
        if input_shape == output_shape:
            allowed_layers.append(all_add[i])

    # The last block is enabled
    if len(all_add) > 0:
        allowed_layers.append(all_add[-1])

    return allowed_layers

def add_to_downsampling(model):
    layers = []
    all_add = []

    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Add):
            all_add.append(i)

    for i in range(1, len(all_add) - 1):
        input_shape = model.get_layer(index=all_add[i]).output_shape
        output_shape = model.get_layer(index=all_add[i - 1]).output_shape
        # These are the downsampling add
        if input_shape != output_shape:
            layers.append(all_add[i])

    return layers

def idx_score_block(blocks, layers):
    # Associates the scores' index with the ResNet block
    output = {}
    idx = 0
    for i in range(0, len(blocks)):
        # blocks[i] is number of residual blocks in stage i
        for layer_idx in range(idx, idx + blocks[i] - 1):
            output[layers[layer_idx]] = i
            idx = idx + 1

    return output

def new_blocks(blocks, scores, allowed_layers, p=0.1):
    num_blocks = list(blocks)  # make a mutable copy

    if isinstance(p, float):
        num_remove = round(p * len(scores))
    else:
        num_remove = int(p)

    score_block = idx_score_block(blocks, allowed_layers)
    mask = np.ones(len(allowed_layers))

    i = num_remove
    # It forces to remove 'num_remove' layers
    while i > 0 and not np.all(np.isinf(scores)):
        min_score = np.argmin(scores)
        layer_at_min = allowed_layers[min_score]
        block_idx = score_block[layer_at_min]

        if num_blocks[block_idx] - 1 > 1:
            mask[min_score] = 0
            num_blocks[block_idx] = num_blocks[block_idx] - 1
            i = i - 1

        scores[min_score] = np.inf

    return num_blocks, mask

def transfer_weightsBN(model, new_model, mask):
    add_model = blocks_to_prune(model)
    add_new_model = blocks_to_prune(new_model)

    # Add the same weights until finding the first Add layer
    for idx in range(0, len(model.layers)):
        w = model.get_layer(index=idx).get_weights()
        # Some layers may have no weights; set_weights expects matching shapes
        if w:
            try:
                new_model.get_layer(index=idx).set_weights(w)
            except Exception:
                # best-effort: skip if shapes mismatch
                pass

        if isinstance(model.get_layer(index=idx), Add):
            break

    # These are the layers where the weights must be transferred
    add_model = np.array(add_model)[mask == 1]
    add_model = list(add_model)
    end = len(add_new_model)

    for layer_idx in range(0, end):
        idx_model = np.arange(add_model[0] - 9, add_model[0] + 1).tolist()
        idx_new_model = np.arange(add_new_model[0] - 9, add_new_model[0] + 1).tolist()

        for transfer_idx in range(0, len(idx_model)):
            w = model.get_layer(index=idx_model[transfer_idx]).get_weights()
            if w:
                try:
                    new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)
                except Exception:
                    pass

        add_new_model.pop(0)
        add_model.pop(0)

    # These are the downsampling layers
    add_model = add_to_downsampling(model)
    add_new_model = add_to_downsampling(new_model)

    for i in range(0, len(add_model)):
        idx_model = np.arange(add_model[i] - 11, add_model[i] + 1).tolist()
        idx_new_model = np.arange(add_new_model[i] - 11, add_new_model[i] + 1).tolist()

        for transfer_idx in range(0, len(idx_model)):
            w = model.get_layer(index=idx_model[transfer_idx]).get_weights()
            if w:
                try:
                    new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)
                except Exception:
                    pass

    # This is the dense layer (assume final layer is last)
    try:
        w = model.get_layer(index=-1).get_weights()
        if w:
            new_model.get_layer(index=-1).set_weights(w)
    except Exception:
        pass

    return new_model

def count_res_blocks
