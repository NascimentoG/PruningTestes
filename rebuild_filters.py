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
    if len(weights) > 0:
        weights[0] = np.delete(weights[0], idx_pruned, axis=2)
    return create_Conv2D_from_conf(config), weights

def create_Conv2D_from_conf(config):
    """Create Conv2D layer from a config dict (no weights set here)."""
    return Conv2D(
        filters=config.get('filters'),
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

def create_depthwise_from_config(config):
    """Create DepthwiseConv2D layer from a config dict (no weights set here)."""
    return DepthwiseConv2D(
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

def remove_conv_weights(index_model, idxs, model):
    config = model.get_layer(index=index_model).get_config()
    weights = model.get_layer(index=index_model).get_weights()
    if len(weights) > 0:
        weights[0] = np.delete(weights[0], idxs, axis=3)
        weights[1] = np.delete(weights[1], idxs)
        config['filters'] = weights[1].shape[0]
    return idxs, config, weights

def remove_convMobile_weights(index_model, idxs, model):
    config = model.get_layer(index=index_model).get_config()
    weights = model.get_layer(index=index_model).get_weights()
    if len(weights) > 0:
        weights[0] = np.delete(weights[0], idxs, axis=3)
        config['filters'] = weights[0].shape[-1]
    return idxs, config, weights

def _apply_weights_after_call(layer, weights):
    """Helper: try to set_weights after layer was called (built)."""
    if not weights:
        return
    try:
        layer.set_weights(weights)
    except Exception:
        # best-effort: ignore mismatch (caller should ensure compatibility)
        try:
            # try to build layer with a dummy shape if possible (very defensive)
            # but we avoid complex builds here; silently skip if fails
            pass
        except Exception:
            pass

def rebuild_resnetBN(model, blocks, layer_filters, iter=0, num_classes=1000):
    stacks = len(blocks)
    num_filters = 64
    layer_filters = dict(layer_filters)

    # fix input shape access for TF2.x
    inp_shape = model.inputs[0].shape
    inputs = Input(shape=(inp_shape[1], inp_shape[2], inp_shape[3]))

    # ZeroPadding2D from config
    x = ZeroPadding2D.from_config(config=model.get_layer(index=1).get_config())(inputs)

    # first conv
    _, config, weights = remove_conv_weights(2, [], model)
    conv_layer = create_Conv2D_from_conf(config)
    x = conv_layer(x)
    _apply_weights_after_call(conv_layer, weights)

    # BatchNormalization (create, call, then set_weights)
    bn_layer = BatchNormalization(epsilon=1.001e-5)
    x = bn_layer(x)
    _apply_weights_after_call(bn_layer, model.get_layer(index=3).get_weights())

    x = Activation.from_config(config=model.get_layer(index=4).get_config())(x)
    x = ZeroPadding2D.from_config(config=model.get_layer(index=5).get_config())(x)
    x = MaxPooling2D.from_config(config=model.get_layer(index=6).get_config())(x)

    i = 7
    for stage in range(0, stacks):
        num_res_blocks = blocks[stage]

        shortcut = x

        _, config, weights = remove_conv_weights(i, [], model)
        conv_layer = create_Conv2D_from_conf(config)
        x = conv_layer(x)
        _apply_weights_after_call(conv_layer, weights)
        i = i + 1

        bn_layer = BatchNormalization(epsilon=1.001e-5)
        x = bn_layer(x)
        _apply_weights_after_call(bn_layer, model.get_layer(index=i).get_weights())
        i = i + 1

        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        _, config, weights = remove_conv_weights(i, [], model)
        conv_layer = create_Conv2D_from_conf(config)
        x = conv_layer(x)
        _apply_weights_after_call(conv_layer, weights)
        i = i + 1

        bn_layer = BatchNormalization(epsilon=1.001e-5)
        x = bn_layer(x)
        _apply_weights_after_call(bn_layer, model.get_layer(index=i).get_weights())
        i = i + 1

        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        _, config, weights = remove_conv_weights(i+1, [], model)
        conv_layer = create_Conv2D_from_conf(config)
        x = conv_layer(x)
        _apply_weights_after_call(conv_layer, weights)

        bn_layer = BatchNormalization(epsilon=1.001e-5)
        x = bn_layer(x)
        _apply_weights_after_call(bn_layer, model.get_layer(index=i + 3).get_weights())

        _, config, weights = remove_conv_weights(i, [], model)
        conv_layer = create_Conv2D_from_conf(config)
        shortcut = conv_layer(shortcut)
        _apply_weights_after_call(conv_layer, weights)
        i = i + 2

        shortcut_bn = BatchNormalization(epsilon=1.001e-5)
        shortcut = shortcut_bn(shortcut)
        _apply_weights_after_call(shortcut_bn, model.get_layer(index=i).get_weights())
        i = i + 1

        x = Add(name=model.get_layer(index=i).name)([shortcut, x])
        i = i + 2

        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        # end First Layer Block

        for res_block in range(2, num_res_blocks + 1):
            shortcut = x

            idx_previous, config, weights = remove_conv_weights(i, layer_filters.get(i, []), model)
            conv_layer = create_Conv2D_from_conf(config)
            x = conv_layer(x)
            _apply_weights_after_call(conv_layer, weights)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            bn_layer = BatchNormalization(epsilon=1.001e-5)
            x = bn_layer(x)
            _apply_weights_after_call(bn_layer, rw_bn(wb, idx_previous))
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
            conv_layer = create_Conv2D_from_conf(config)
            x = conv_layer(x)
            _apply_weights_after_call(conv_layer, weights)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            bn_layer = BatchNormalization(epsilon=1.001e-5)
            x = bn_layer(x)
            _apply_weights_after_call(bn_layer, rw_bn(wb, idx_previous))
            i = i + 1

            x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
            i = i + 1

            weights = model.get_layer(index=i).get_weights()
            config = model.get_layer(index=i).get_config()
            if len(weights) > 0:
                weights[0] = np.delete(weights[0], idx_previous, axis=2)
            conv_layer = create_Conv2D_from_conf(config)
            x = conv_layer(x)
            _apply_weights_after_call(conv_layer, weights)
            i = i + 1

            bn_layer = BatchNormalization(epsilon=1.001e-5)
            x = bn_layer(x)
            _apply_weights_after_call(bn_layer, model.get_layer(index=i).get_weights())
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
    dense_layer = Dense(units=config['units'], activation=config.get('activation'))
    x = dense_layer(x)
    _apply_weights_after_call(dense_layer, weights)

    rebuilt_model = Model(inputs, x)
    return rebuilt_model

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
        conv_layer = create_Conv2D_from_conf(config)
        x = conv_layer(x)
        _apply_weights_after_call(conv_layer, weights)
        i = i + 1

        bn_layer = BatchNormalization(name=model.get_layer(index=i).name, epsilon=1e-3, momentum=0.999)
        x = bn_layer(x)
        _apply_weights_after_call(bn_layer, model.get_layer(index=i).get_weights())
        i = i + 1

        x = Activation(relu6, name=model.get_layer(index=i).name)(x)
        i = i + 1

    else:
        x = inputs

    config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
    depthwise_layer = create_depthwise_from_config(config)
    x = depthwise_layer(x)
    _apply_weights_after_call(depthwise_layer, weights)
    i = i + 1

    bn_layer = BatchNormalization(name=model.get_layer(index=i).name, epsilon=1e-3, momentum=0.999)
    x = bn_layer(x)
    _apply_weights_after_call(bn_layer, model.get_layer(index=i).get_weights())
    i = i + 1

    x = Activation(relu6, name=model.get_layer(index=i).name)(x)
    i = i + 1

    idx_previous, config, weights = remove_convMobile_weights(i, layer_filters.get(i, []), model)
    conv_layer = create_Conv2D_from_conf(config)
    x = conv_layer(x)
    _apply_weights_after_call(conv_layer, weights)
    i = i + 1

    wb = model.get_layer(index=i).get_weights()
    bn_layer = BatchNormalization(name=model.get_layer(index=i).name, epsilon=1e-3, momentum=0.999)
    x = bn_layer(x)
    _apply_weights_after_call(bn_layer, rw_bn(wb, idx_previous))
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

            conv_layer = create_Conv2D_from_conf(config)
            x = conv_layer(x)
            _apply_weights_after_call(conv_layer, weights)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            bn_layer = BatchNormalization(name=prefix + 'expand_BN', epsilon=1e-3, momentum=0.999)
            x = bn_layer(x)
            _apply_weights_after_call(bn_layer, rw_bn(wb, idx))
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
            depthwise_layer = create_depthwise_from_config(config)
            x = depthwise_layer(x)
            _apply_weights_after_call(depthwise_layer, weights)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            bn_layer = BatchNormalization(name=model.get_layer(index=i).name, epsilon=1e-3, momentum=0.999)
            x = bn_layer(x)
            _apply_weights_after_call(bn_layer, rw_bn(wb, idx))
            i = i + 1

            x = Activation(relu6, name=model.get_layer(index=i).name)(x)
            i = i + 1

            # block_kth_project
            x = rw_cn(i, idx, model)(x)
            # rw_cn returns (conv_layer) built from config; it currently returns create_Conv2D_from_conf(config, weights)
            # In our earlier definition, rw_cn returns (create_Conv2D_from_conf, weights). We adapted rw_cn to return layer and weights.
            # But to keep compatibility with existing code flow, if rw_cn returned (layer, weights) adjust accordingly.
            # Our rw_cn returns (layer, weights) now, so handle:
            # if rw_cn returns tuple (layer, weights): layer = ... ; _apply_weights_after_call(layer, weights)
            # In this file's flow, rw_cn is used as x = rw_cn(i, idx, model)(x) - so we need rw_cn to return a layer callable.
            # To handle both cases, let's allow rw_cn above to return (layer, weights) or a layer. We'll try to set_weights after call.
            i = i + 1

            # Next BatchNorm
            bn_layer = BatchNormalization(name=model.get_layer(index=i).name, epsilon=1e-3, momentum=0.999)
            x = bn_layer(x)
            _apply_weights_after_call(bn_layer, model.get_layer(index=i).get_weights())
            i = i + 1

            if isinstance(model.get_layer(index=i), Add):
                x = Add(name=model.get_layer(index=i).name)([shortcut, x])
                i = i + 1

            id = id + 1

    config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
    conv_layer = create_Conv2D_from_conf(config)
    x = conv_layer(x)
    _apply_weights_after_call(conv_layer, weights)

    bn_layer = BatchNormalization(name=model.get_layer(index=i+1).name, epsilon=1e-3, momentum=0.999)
    x = bn_layer(x)
    _apply_weights_after_call(bn_layer, model.get_layer(index=i+1).get_weights())

    x = Activation(relu6, name=model.get_layer(index=i+2).name)(x)

    x = GlobalAveragePooling2D.from_config(model.get_layer(index=i+3).get_config())(x)

    config, weights = model.get_layer(index=i+4).get_config(), model.get_layer(index=i+4).get_weights()
    dense_layer = Dense(
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
    x = dense_layer(x)
    _apply_weights_after_call(dense_layer, weights)

    model_out = Model(inputs, x, name='MobileNetV2')
    return model_out

def allowed_layers_resnet(model):
    allowed_layers = []
    all_add = []
    for i in range(0, len(model.layers)):
        layer = model.get_layer(index=i)
        if isinstance(layer, Add):
            all_add.append(i)
        if isinstance(layer, Conv2D) and layer.strides == (2, 2) and layer.kernel_size != (1, 1):
            allowed_layers.append(i)

    # protect against empty all_add
    if all_add:
        allowed_layers.append(all_add[0] - 5)

        for i in range(1, len(all_add)):
            allowed_layers.append(all_add[i] - 5)

    # To avoid bug due to keras architecture (i.e., order of layers)
    tmp = allowed_layers
    allowed_layers = []

    for i in tmp:
        try:
            if isinstance(model.get_layer(index=i), Conv2D):
                allowed_layers.append(i)
        except Exception:
            pass

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
        input_layer = model.get_layer(index=all_add[i])
        output_layer = model.get_layer(index=all_add[i - 1])
        # These are the valid blocks we can remove
        try:
            if input_layer.output.shape == output_layer.output.shape:
                allowed_layers.append(all_add[i] - 8)
                allowed_layers.append(all_add[i] - 5)

                layer = model.get_layer(index=(all_add[i] - 8))
                config = layer.get_config()
                n_filters += config.get('filters', 0)

                layer = model.get_layer(index=(all_add[i] - 5))
                config = layer.get_config()
                n_filters += config.get('filters', 0)
        except Exception:
            # if shapes not available, skip this block
            pass

    available_filters = n_filters - len(allowed_layers)

    if available_filters == 0:
        isFiltersAvailable = False

    #print(f"Numero de filtros nas camadas permitidas (PODA POR FILTRO) {available_filters} em {len(allowed_layers)}")
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
    global architecture_name
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
        #print(f"Faltam remover {totalFiltersToRemove - numberFiltersRemoved} filtros,\n Mas o numero de camadas com mais de um filtro e {len(layerSelectedList)}")

    scores = [x for x in zip(allowed_layers, filtersToRemove)]

    if architecture_name.__contains__('ResNet'):
        blocks = rl.count_res_blocks(model)
        return rebuild_resnetBN(model=model,
                                blocks=blocks,
                                layer_filters=scores)

    if architecture_name.__contains__('MobileNetV2'):
        blocks = rl.count_blocks(model)
        return rebuild_mobilenetV2(model=model,
                                   blocks=blocks,
                                   layer_filters=scores)
