import tensorflow as tf

import tensorflow_tools as tft


def build_conv_capsuels(input_vectors, num_of_convs, capsule_depth, filter_size, stride, name=""):
    """
    Purpose:
        Apply a convolutional layer to the input vectors and output the vectors.
    Precondition:
        :param input_vectors: The vectors to apply the convolution on. Must be a 4 dimensional tensor with dimension 0 being None.
        :param num_of_convs: Number of convolutional layers to apply.
        :param capsule_depth: The depth of the capsule in the output capsules.
        :param filter_size: Size of the filters.
        :param stride: The stride of the filters.
    Postcondition:
        :return: A 5d tensor: [None, num_of_convs, numb_caps_y, num_caps_x, capsule_depth]
    """
    input_vectors = tf.expand_dims(input_vectors, axis=0)
    input_vectors = tf.tile(input_vectors, [num_of_convs, 1, 1, 1, 1])
    vector_depth = input_vectors.get_shape().as_list()[-1]
    output_vectors = tf.map_fn(lambda x: tft.build_conv_lambda(input=x,
                                                               num_input_channels=vector_depth,
                                                               filter_size=filter_size,
                                                               num_filters=capsule_depth,
                                                               strides=[1, stride, stride, 1],
                                                               use_pooling=False,
                                                               name=name + ".looped_conv",
                                                               padding='VALID'),
                               input_vectors)
    return tf.transpose(output_vectors, perm=[1,0,2,3,4])
    """
    vector_depth = input_vectors.get_shape().as_list()[-1]
    capsuels = []
    for i in range(num_of_convs):
        conv_caps = tft.build_conv_lambda(input=input_vectors,
                              num_input_channels=vector_depth,
                              filter_size=filter_size,
                              num_filters=capsule_depth,
                              strides=[1, stride, stride, 1],
                              use_pooling=False,
                              name=name + ".looped_conv",
                              padding='VALID')
        capsuels.append(tf.expand_dims(conv_caps, axis=1))
    return tf.concat(capsuels, axis=1)
    """


def flatten_capsules(input_vectors):
    """
    Purpose:
        To flatten a capsule layer.
    Precondition:
        :param input_vectors: Capsules to flatten. Dimension 0 must be None.
    Postcondition:
        :return: A flattened capsule layer with shape: [None, num_capsules, capsule_depth]
    """
    capsule_shape = input_vectors.get_shape()
    num_capsules = capsule_shape[1:len(list(capsule_shape)) - 1].num_elements()
    return tf.reshape(input_vectors, shape=[-1, num_capsules, capsule_shape.as_list().pop()])


def route_with_softmax(capsule_hats, num_routes):
    shape = capsule_hats.get_shape().as_list()
    coupling_coef_logits = tf.zeros(shape, dtype=tf.float32)
    for r in range(num_routes):
        coupling_coef = tf.nn.softmax(coupling_coef_logits, dim=0)
        weighted_sum = tf.multiply(coupling_coef, capsule_hats)
        vector_output = tft.squashing_sigmoid(weighted_sum)
        coupling_coef_logits = tf.add(coupling_coef_logits, tf.multiply(capsule_hats, vector_output))

        # coupling_coef_logits = tf.zeros(coupling_coef_shape, dtype=tf.float32)

    return vector_output


def build_flat_capsule_layer(input_capsules, num_routes, num_outputs, output_depth, routing_func):
    """
    Purpose:
        Building a flat capsule layer with a given routing function.
    Precondition:
        :param input_capsules: The capsules to be routed.
        :param num_routes: The number of routing iterations to do.
        :param num_outputs: The number of capsul
        :param output_depth: The depth of the output capsule layer.
        :param routing_func: The routing function to use.
    Postcondition:
        :return: The vector outputs of the routed capsules.
    """
    original_shape = input_capsules.get_shape().as_list()
    reshaped = False
    if len(original_shape) == 4:
        reshaped = True
        input_capsules = tf.reshape(input_capsules, shape=[-1, original_shape[1] * original_shape[2], original_shape[3]])

    input_capsules = tf.expand_dims(input_capsules, axis=-2)
    capsule_shape = input_capsules.get_shape().as_list()
    weight_shape = [capsule_shape[1], capsule_shape[-1], output_depth]
    capsule_hats = []
    for i in range(num_outputs):
        weights = tft.new_weights(shape=weight_shape)
        one_output = tf.map_fn(lambda x: tf.matmul(x, weights), input_capsules)
        capsule_hats.append(tf.expand_dims(one_output, axis=1))
    capsule_hats = tf.concat(capsule_hats, axis=1)
    if reshaped:
        capsule_hats = tf.reshape(capsule_hats, shape=[-1, num_outputs, original_shape[1], original_shape[2], output_depth])
    else:
        capsule_hats = tf.reduce_sum(capsule_hats, axis=3)
    output_vectors = tf.map_fn(lambda x: routing_func(x, num_routes), capsule_hats)
    return tf.reduce_sum(output_vectors, axis=-2)


def build_layered_conv_capsules(input_vectors, num_of_convs, capsule_depth, filter_size, stride, name=""):
    """
    Purpose:
        Layer convolutional capsule layers on convolutional capsule layers.
    Precondition:
        :param input_vectors: The vectors to apply the convolution on. Must be a 5 dimensional tensor with dimension 0 being None.
        :param num_of_convs: Number of convolutional layers to apply.
        :param capsuel_depth: The depth of the capsule in the output capsules.
        :param filter_size: Size of the filters.
        :param stride: The stride of the filters.
    Postcondition:
        :return: A 6d tensor: [None, num_of_convs, numb_caps_y, num_caps_x, capsule_depth]
    """
    input_vectors = tf.transpose(input_vectors, perm=[1, 0, 2, 3, 4])
    output_vectors = tf.map_fn(lambda x: build_conv_capsuels(x, num_of_convs, capsule_depth, filter_size, stride, name), input_vectors, dtype=tf.float32)
    output_vectors = tf.transpose(output_vectors, perm=[1, 0, 2, 3, 4, 5])
    output_shape = output_vectors.get_shape().as_list()
    return tf.reshape(output_vectors, shape=[-1, output_shape[1] * output_shape[2], output_shape[3], output_shape[4], output_shape[5]])
