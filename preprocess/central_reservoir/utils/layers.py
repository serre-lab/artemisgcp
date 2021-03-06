import tensorflow as tf

from . import cross_replica_batch_norm
#import tpu_normalization
#import crbn

#TODO: 1.maxpool3d, averagepool2d/3d

def linear(input_var, layer_name, output_units,
        activation=tf.keras.activations.relu,
        initializer=tf.keras.initializers.glorot_normal,
        batch_norm=False,
        update_collection=False,
        is_training=True,
        bias=False):

    # Get shape of a tensor as a list
    shape = input_var.get_shape().as_list()

    with tf.variable_scope(
            layer_name):
 
        weight = tf.get_variable(
            "w", 
            [shape[-1], output_units],
            tf.float32,
            initializer=initalizer)

        output_var = tf.matmul(
            input_var,
            weight)

        if not batch_norm:
            bias_var = tf.get_variable(
                "b",
                [output_units],
                initializer=tf.constant_initalizer(0.0))

            output_var = tf.nn.bias_add(
                output_var,
                bias_var)

        #TODO: Incorporate batch norm layer

        return activation(output_var)


def conv_batchnorm_relu(input_var, layer_name, out_channels,
        activation=tf.keras.activations.relu,
        kernel_size=1,
        stride=1,
        padding='SAME',
        spectral_norm_flag=False,
        update_collection=False,
        is_training=True,
        use_batch_norm=False,
        use_cross_replica_batch_norm=False,
        num_cores=8,
        initializer=tf.keras.initializers.glorot_normal):

    shape = input_var.get_shape().as_list()
    in_channels = shape[-1]
    k = kernel_size
    s = stride

    if len(shape) == 4:
        # :input_var: [batch, height, width, in_channels]
        filter_shape = [k, k, in_channels, out_channels]
        strides = [1, s, s, 1]
        conv_name = 'conv_2d'

    elif len(shape) == 5:
        # :input_var: [batch, depth, height, width, in_channels]
        filter_shape = [k, k, k, in_channels, out_channels]
        strides = [1, s, s, s, 1]
        conv_name = 'conv_3d'

    with tf.variable_scope(
        layer_name):

        w = tf.get_variable(
            'w',
            filter_shape,
            initializer=initializer)
  
        if len(shape) == 5:
            conv = tf.nn.conv3d(
                input_var,
                w,
                strides,
                padding,
                name=conv_name)

        elif len(shape) == 4:
            conv = tf.nn.conv2d(
                input_var,
                w,
                strides,
                padding,
                name=conv_name)

        if use_batch_norm:
            if use_cross_replica_batch_norm:
                conv = tpu_normalization.cross_replica_batch_normalization(
                    inputs=conv,
                    axis=-1,
                    training=is_training,
                    center=True,
                    scale=True,
                    num_distributed_groups=1)
            else:
                conv = tf.layers.batch_normalization(
                    conv,
                    scale=True,
                    center=True,
                    training=is_training,
                    name='batch_normalization')

        else:
            bias_var = tf.get_variable(
                'b',
                [out_channels],
                initializer=tf.constant_initializer(0.0))

            conv = tf.nn.bias_add(
                conv,
                bias_var)

        if activation is None:
            return conv
        else:
            return activation(conv)

def upconv_2D(input_var, layer_name, n_filters,
                kernel_size=(2, 2), strides=(2, 2),
                use_bias=True, padding='valid'):

    '''Up convolution tensor
    Args:
        input_var: Tensor (N, H, W, C) representing input
        layer_name: String representing name of layer for scoping
        n_filters: Int to specify number of output filters
        kernel_size: Tuple of 2 Ints to specify spatial
                dimension of filters
        strides: Tuple of 2 Ints to specify strides of the
                convolution
        use_bias: Boolean to specify whether to use Bias
        padding: one of 'valid' or 'same'

    Returns: 
        4-D tensor: 
    '''

    ks_type = type(kernel_size)
    st_type = type(strides)
    inp_shape = input_var.get_shape().as_list()

    assert ks_type in [list, tuple],\
        'kernel_size type mismatch, should be tuple or list'
    assert st_type in [list, tuple],\
        'strides type mismatch, should be tuple or list'
    assert len(inp_shape) == 4,\
        'Input tensor shape length should be 4, found %d'\
            %len(inp_shape)

    with tf.variable_scope(
        layer_name):

        upconv = tf.layers.conv2d_transpose(
            input_var,
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding=padding,
            name='upconv')

        return upconv


def maxpool(input_var, layer_name, ksize,
    strides, padding='SAME'):

    shape = input_var.get_shape().as_list()

    if len(shape) == 4:
        # :input_var: [batch, height, width, in_channels]
        net = tf.nn.max_pool(
            input_var,
            ksize=ksize,
            strides=strides,
            padding=padding,
            name=layer_name)

    elif len(shape) == 5:
        # :input_var: [batch, depth, height, width, in_channels]
        net = tf.nn.max_pool3d(
            input_var,
            ksize=ksize,
            strides=strides,
            padding=padding,
            name=layer_name)

    return net

def avgpool(input_var, ksize, strides,
    padding='SAME'):

    shape = input_var.get_shape().as_list()

    if len(shape) == 4:
        # :input_var: [batch, height, width, in_channels]
        net = tf.nn.avg_pool2d(
            input_var,
            ksize=ksize,
            strides=strides,
            padding=padding)

    elif len(shape) == 5:
        # :input_var: [batch, depth, height, width, in_channels]
        net = tf.nn.avg_pool3d(
            input_var,
            ksize=ksize,
            strides=strides,
            padding=padding)

    return net
