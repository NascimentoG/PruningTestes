import numpy as np
import random
import copy
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import rebuild_layers as rl  # It implements some particular functions we need to use here

isFiltersAvailable = True

# Used by MobileNet
def relu6(x):
    return K.relu(x, max_value=6)

def rw_bn(w, index):
    w[0] = np.delete(w[0], index)
    w[1] = np.delete(w[1], index)
    w[2] = np.delete(w[2], index)
    w[3] = np.delete(w[3], index)
    return w

def rw_cn(index_model, idx_pruned, model):
    # This function removes the weights of the Conv2D considering the previous pruning in other Conv2D
    config = model.get_layer(index=index_model).get_config()
    weights = model.get_layer(index=index_model).get_weights()
    weights[0] = np.delete(weights[0], idx_pruned, axis=2)
    return create_Conv2D_from_conf(config, weights)

def create_Conv2D_from_conf(config, weights=None):
    """
    Create Conv2D using the layer config and, if provided, set the weights using set_weights.
    """
    n_filters = None
    if weights is not None and len(weights) > 0:
        n_filters = weights[0].shape[-1]

    conv = Conv2D(
        filters=n_filters if n_filters is not None else config.get('filters'),
        kernel_size=config.get('kernel_size'),
        strides=config.get('strides'),
        padding=config.get('padding'),
        data_format=config.get('data_format'),
        dilation_rate=config.get('dilation_rate'),
        activation=config.get('activation'),
        use_bias=config.get('use_bias'),
        kernel_constraint=config.get('kernel_constraint'),
        bias_constraint=config.get('bias_constraint'),
        kernel_regularizer=config.get('kernel_regularizer'),
        bias_regularizer=config.get('bias_regularizer'),
        activity_regularizer=config.get('activity_regularizer'),
        name=config.get('name'),
        trainable=config.get('trainable')
    )

    if weights:
        try:
            conv.set_weights(weights)
        except Exception:
            # In case the weights shape differs a bit, still attach what is possible
            conv.build((None, None, None, weights[0].shape[2]))
            conv.set_weights(weights)
    return conv

def create_depthwise_from_config(config, weights=None):
    depthwise = DepthwiseConv2D(
        kernel_size=config.get('kernel_size'),
        strides=config.get('strides'),
        padding=config.get('padding'),
        data_format=config.get('data_format'),
        dilation_rate=config.get('dilation_rate'),
        depth_multiplier=config.get('depth_multiplier'),
        activation=config.get('activation'),
        use_bias=config.get('use_bias'),
        depthwise_constraint=config.get('depthwise_constraint'),
        depthwise_regularizer=config.get('depthwise_regularizer'),
        depthwise_initializer=config.get('depthwise_initializer'),
        name=config.get('name'),
        trainable=config.get('trainable')
    )

    if weights:
        try:
            depthwise.set_weights(weights)
        except Exception:
            # best-effort build then set
            depthwise.build((None, None, None, weights[0].shape[2]))
            depthwise.set_weights(weights)
    return depthwise

def remove_conv_weights(index_model, idxs, model):
    config, weights = (model.get_layer(index=index_model).get_config(),
                       model.get_layer(index=index_model).get_weights())
    if len(weights) > 0:
        weights[0] = np.delete(weights[0], idxs, axis=3)
        weights[1] = np.delete(weights[1], idxs)
        config['filters'] = weights[1].shape[0]
    return idxs, config, weights

def remove_convMobile_weights(index_model, idxs, model):
    config, weights = (model.get_layer(index=index_model).get_config(),
                       model.get_layer(index=index_model).get_weights())
    if len(weights) > 0:
        weights[0] = np.delete(weights[0], idxs, axis=3)
        config['filters'] = weights[0].shape[-1]
    return idxs, config, weights

def rebuild_resnetBN(model, blocks, layer_filters, iter=0, num_classes=1000):
    stacks = len(blocks)
    num_filters = 64
    layer_filters = dict(layer_filters)

    inp_shape = model.inputs[0].shape
    inputs = Input(shape=(inp_shape[1], inp_shape[2], inp_shape[3]))

    # ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = ZeroPadding2D.from_config(config=model.get_layer(index=1).get_config())(inputs)

    # Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(x)
    _, config, weights = remove_conv_weights(2, [], model)
    conv = create_Conv2D_from_conf(config, weights)
    x = conv(x)

    # BatchNormalization
    bn_weights = model.get_layer(index=3).get_weights()
    bn = BatchNormalization(epsilon=1.001e-5)
    if bn_weights:
        bn.set_weights(bn_weights)
    x = bn(x)

    x = Activation.from_config(config=model.get_layer(index=4).get_config())(x)

    x = ZeroPadding2D.from_config(config=model.get_layer(index=5).get_config())(x)

    x = MaxPooling2D.from_config(config=model.get_layer(index=6).get_config())(x)

    i = 7
    for stage in range(0, stacks):
        num_res_blocks = blocks[stage]

        shortcut = x

        _, config, weights = remove_conv_weights(i, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        x = conv(x)
        i = i + 1

        bn_weights = model.get_layer(index=i).get_weights()
        bn = BatchNormalization(epsilon=1.001e-5)
        if bn_weights:
            bn.set_weights(bn_weights)
        x = bn(x)
        i = i + 1

        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        _, config, weights = remove_conv_weights(i, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        x = conv(x)
        i = i + 1

        bn_weights = model.get_layer(index=i).get_weights()
        bn = BatchNormalization(epsilon=1.001e-5)
        if bn_weights:
            bn.set_weights(bn_weights)
        x = bn(x)
        i = i + 1

        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        _, config, weights = remove_conv_weights(i+1, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        x = conv(x)

        bn_weights = model.get_layer(index=i + 3).get_weights()
        bn = BatchNormalization(epsilon=1.001e-5)
        if bn_weights:
            bn.set_weights(bn_weights)
        x = bn(x)

        _, config, weights = remove_conv_weights(i, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        shortcut = conv(shortcut)
        i = i + 2

        bn_weights = model.get_layer(index=i).get_weights()
        bn = BatchNormalization(epsilon=1.001e-5)
        if bn_weights:
            bn.set_weights(bn_weights)
        shortcut = bn(shortcut)
        i = i + 1

        x = Add(name=model.get_layer(index=i).name)([shortcut, x])
        i = i + 2

        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        # end First Layer Block

        for res_block in range(2, num_res_blocks + 1):
            shortcut = x

            idx_previous, config, weights = remove_conv_weights(i, layer_filters.get(i, []), model)
            conv = create_Conv2D_from_conf(config, weights)
            x = conv(x)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            bn = BatchNormalization(epsilon=1.001e-5)
            if wb:
                bn.set_weights(rw_bn(wb, idx_previous))
            x = bn(x)
            i = i + 1

            x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
            i = i + 1

            weights = model.get_layer(index=i).get_weights()
            config = model.get_layer(index=i).get_config()
            idxs = layer_filters.get(i, [])
            if len(weights) > 0:
                weights[0] = np.delete(weights[0], idxs, axis=3)
                weights[1] = np.delete(weights[1], idxs)
                weights[0] = np.delete(weights[0], idx_previous, axis=2)
            idx_previous = idxs
            conv = create_Conv2D_from_conf(config, weights)
            x = conv(x)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            bn = BatchNormalization(epsilon=1.001e-5)
            if wb:
                bn.set_weights(rw_bn(wb, idx_previous))
            x = bn(x)
            i = i + 1

            x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
            i = i + 1

            weights = model.get_layer(index=i).get_weights()
            config = model.get_layer(index=i).get_config()
            if len(weights) > 0:
                weights[0] = np.delete(weights[0], idx_previous, axis=2)
            conv = create_Conv2D_from_conf(config, weights)
            x = conv(x)
            i = i + 1

            bn_weights = model.get_layer(index=i).get_weights()
            bn = BatchNormalization(epsilon=1.001e-5)
            if bn_weights:
                bn.set_weights(bn_weights)
            x = bn(x)
            i = i + 1

            x = Add.from_config(config=model.get_layer(index=i).get_config())([shortcut, x])
            i = i + 1

            x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
            i = i + 1

        num_filters = num_filters * 2

    x = GlobalAveragePooling2D.from_config(config=model.get_layer(index=i).get_config())(x)
    i = i + 1

    weights = model.get_layer(index=i).get_weights()
    config = model.get_layer(index=i).get_config()
    dense = Dense(units=config['units'], activation=config.get('activation'))
    if weights:
        dense.set_weights(weights)
    x = dense(x)

    model = Model(inputs, x)
    return model

def rebuild_mobilenetV2(model, blocks, layer_filters, initial_reduction=False, num_classes=1000):
    blocks = np.append(blocks, 1)
    stacks = len(blocks)
    layer_filters = dict(layer_filters)

    inp_shape = model.inputs[0].shape
    inputs = Input(shape=(inp_shape[1], inp_shape[2], inp_shape[3]))

    idx_previous = []
    i = 1
    if isinstance(model.get_layer(index=i), ZeroPadding2D):
        x = ZeroPadding2D.from_config(model.get_layer(index=i).get_config())(inputs)
        i = i + 1

        config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
        conv = create_Conv2D_from_conf(config, weights)
        x = conv(x)
        i = i + 1

        bn_weights = model.get_layer(index=i).get_weights()
        bn = BatchNormalization(name=model.get_layer(index=i).name, epsilon=1e-3, momentum=0.999)
        if bn_weights:
            bn.set_weights(bn_weights)
        x = bn(x)
        i = i + 1

        x = Activation(relu6, name=model.get_layer(index=i).name)(x)
        i = i + 1

    else:
        x = inputs

    config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
    depthwise = create_depthwise_from_config(config, weights)
    x = depthwise(x)
    i = i + 1

    bn_weights = model.get_layer(index=i).get_weights()
    bn = BatchNormalization(name=model.get_layer(index=i).name, epsilon=1e-3, momentum=0.999)
    if bn_weights:
        bn.set_weights(bn_weights)
    x = bn(x)
    i = i + 1

    x = Activation(relu6, name=model.get_layer(index=i).name)(x)
    i = i + 1

    idx_previous, config, weights = remove_convMobile_weights(i, layer_filters.get(i, []), model)
    conv = create_Conv2D_from_conf(config, weights)
    x = conv(x)
    i = i + 1

    wb = model.get_layer(index=i).get_weights()
    bn = BatchNormalization(name=model.get_layer(index=i).name, epsilon=1e-3, momentum=0.999)
    if wb:
        bn.set_weights(rw_bn(wb, idx_previous))
    x = bn(x)
    i = i + 1

    id = 1
    for stage in range(0, stacks):
        num_blocks = blocks[stage]

        for mobile_block in range(0, num_blocks):
            prefix = 'block_{}_'.format(id)
            shortcut = x

            # 1x1 Convolution -- _expand
            idx = layer_filters.get(i, [])
            config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
            if len(weights) > 0:
                weights[0] = np.delete(weights[0], idx, axis=3)

            # First block expand only
            if id == 1:
                if len(weights) > 0:
                    weights[0] = np.delete(weights[0], idx_previous, axis=2)

            conv = create_Conv2D_from_conf(config, weights)
            x = conv(x)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            bn = BatchNormalization(name=prefix + 'expand_BN', epsilon=1e-3, momentum=0.999)
            if wb:
                bn.set_weights(rw_bn(wb, idx))
            x = bn(x)
            i = i + 1

            x = Activation(relu6, name=model.get_layer(index=i).name)(x)
            i = i + 1

            if isinstance(model.get_layer(index=i), ZeroPadding2D):  # stride==2
                x = ZeroPadding2D.from_config(model.get_layer(index=i).get_config())(x)
                i = i + 1

            # block_kth_depthwise
            config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
            if len(weights) > 0:
                weights[0] = np.delete(weights[0], idx, axis=2)
            depthwise = create_depthwise_from_config(config, weights)
            x = depthwise(x)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            bn = BatchNormalization(name=model.get_layer(index=i).name, epsilon=1e-3, momentum=0.999)
            if wb:
                bn.set_weights(rw_bn(wb, idx))
            x = bn(x)
            i = i + 1

            x = Activation(relu6, name=model.get_layer(index=i).name)(x)
            i = i + 1

            # block_kth_project
            x = rw_cn(i, idx, model)(x)
            i = i + 1

            bn_weights = model.get_layer(index=i).get_weights()
            bn = BatchNormalization(name=model.get_layer(index=i).name, epsilon=1e-3, momentum=0.999)
            if bn_weights:
                bn.set_weights(bn_weights)
            x = bn(x)
            i = i + 1

            if isinstance(model.get_layer(index=i), Add):
                x = Add(name=model.get_layer(index=i).name)([shortcut, x])
                i = i + 1

            id = id + 1

    config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
    conv = create_Conv2D_from_conf(config, weights)
    x = conv(x)

    bn_weights = model.get_layer(index=i+1).get_weights()
    bn = BatchNormalization(name=model.get_layer(index=i+1).name, epsilon=1e-3, momentum=0.999)
    if bn_weights:
        bn.set_weights(bn_weights)
    x = bn(x)

    x = Activation(relu6, name=model.get_layer(index=i+2).name)(x)

    x = GlobalAveragePooling2D.from_config(model.get_layer(index=i+3).get_config())(x)

    config, weights = model.get_layer(index=i+4).get_config(), model.get_layer(index=i+4).get_weights()
    dense = Dense(
        units=config['units'],
        activation=config.get('activation'),
        activity_regularizer=config.get('activity_regularizer'),
        kernel_constraint=config.get('kernel_constraint'),
        kernel_regularizer=config.get('kernel_regularizer'),
        name=config.get('name'),
        trainable=config.get('trainable'),
        use_bias=config.get('use_bias'),
        bias_constraint=config.get('bias_constraint'),
        bias_regularizer=config.get('bias_regularizer')
    )
    if weights:
        dense.set_weights(weights)
    x = dense(x)

    model = Model(inputs, x, name='MobileNetV2')
    return model

def allowed_layers_resnet(model):
    allowed_layers = []
    all_add = []
    for i in range(0, len(model.layers)):
        layer = model.get_layer(index=i)
        if isinstance(layer, Add):
            all_add.append(i)
        if isinstance(layer, Conv2D) and layer.strides == (2, 2) and layer.kernel_size != (1, 1):
            allowed_layers.append(i)

    if all_add:
        allowed_layers.append(all_add[0] - 5)

    for i in range(1, len(all_add)):
        allowed_layers.append(all_add[i] - 5)

    # To avoid bug due to keras architecture (i.e., order of layers)
    # This ensure that only Conv2D are "allowed layers"
    tmp = allowed_layers
    allowed_layers = []

    for i in tmp:
        if isinstance(model.get_layer(index=i), Conv2D):
            allowed_layers.append(i)

    return allowed_layers

def allowed_layers_resnetBN(model):
    global isFiltersAvailable

    allowed_layers = []
    all_add = []
    n_filters = 0
    available_filters = 0

    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Add):
            all_add.append(i)

    for i in range(1, len(all_add) - 1):
        input_shape = model.get_layer(index=all_add[i])
        output_shape = model.get_layer(index=all_add[i - 1])
        # These are the valid blocks we can remove
        if input_shape.output_shape == output_shape.output_shape:
            allowed_layers.append(all_add[i] - 8)
            allowed_layers.append(all_add[i] - 5)

            layer = model.get_layer(index=(all_add[i] - 8))
            config = layer.get_config()
            n_filters += config.get('filters', 0)

            layer = model.get_layer(index=(all_add[i] - 5))
            config = layer.get_config()
            n_filters += config.get('filters', 0)

    available_filters = n_filters - len(allowed_layers)

    if available_filters == 0:
        isFiltersAvailable = False

    print(f"Numero de filtros nas camadas permitidas (PODA POR FILTRO) {available_filters} em {len(allowed_layers)}")
    return allowed_layers

def allowed_layers_mobilenetV2(model):
    allowed_layers = []
    for i in range(0, len(model.layers)):
        layer = model.get_layer(index=i)

        if isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
            if 'expand' in layer.name:
                allowed_layers.append(i)

    return allowed_layers

def idx_to_conv2Didx(model, indices):
    # Convert index onto Conv2D index (required by pruning methods)
    idx_Conv2D = 0
    output = []
    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Conv2D):
            if i in indices:
                output.append(idx_Conv2D)
            idx_Conv2D = idx_Conv2D + 1

    return output

def layer_to_prune_filters(model):
    if architecture_name.__contains__('ResNet'):
        if architecture_name.__contains__('50'):  # ImageNet architectures (ResNet50, 101 and 152)
            allowed_layers = allowed_layers_resnetBN(model)
        else:  # CIFAR-like architectures (low-resolution datasets)
            allowed_layers = allowed_layers_resnet(model)

    if architecture_name.__contains__('MobileNetV2'):
        allowed_layers = allowed_layers_mobilenetV2(model)

    return allowed_layers

def rebuild_network(model, scores, p_filter, totalFiltersToRemove=0, wasPfilterZero=False):
    global isFiltersAvailable
    numberFiltersRemoved = 0
    scores = sorted(scores, key=lambda x: x[0])

    allowed_layers = [x[0] for x in scores]
    scores = [x[1] for x in scores]
    filtersToRemove = copy.deepcopy(scores)

    for i in range(0, len(scores)):
        num_remove = round(p_filter * len(scores[i]))
        numberFiltersRemoved += num_remove
        filtersToRemove[i] = np.argpartition(scores[i], num_remove)[:num_remove]

    layerSelectedList = [i for i in range(0, len(scores))]
    if totalFiltersToRemove != 0 and not wasPfilterZero:
        while ((totalFiltersToRemove - numberFiltersRemoved) != 0) and (len(layerSelectedList) != 0):
            layerSelected = random.choice(layerSelectedList)
            if (len(scores[layerSelected]) - (len(filtersToRemove[layerSelected])) - 1) > 0:
                filterToRemove = np.argpartition(scores[layerSelected], (len(filtersToRemove[layerSelected]) + 1))[:(len(filtersToRemove[layerSelected]) + 1)]
                filtersToRemove[layerSelected] = filterToRemove
                numberFiltersRemoved += 1
            else:
                layerSelectedList.remove(layerSelected)

    if len(layerSelectedList) == 0:
        isFiltersAvailable = False
        print(f"Faltam remover {totalFiltersToRemove - numberFiltersRemoved} filtros,\n Mas o numero de camadas com mais de um filtro e {len(layerSelectedList)}")

    scores = [x for x in zip(allowed_layers, filtersToRemove)]

    if architecture_name.__contains__('ResNet'):
        blocks = rl.count_res_blocks(model)
        return rebuild_resnetBN(model=model, blocks=blocks, layer_filters=scores)

    if architecture_name.__contains__('MobileNetV2'):
        blocks = rl.count_blocks(model)
        return rebuild_mobilenetV2(model=model, blocks=blocks, layer_filters=scores)
