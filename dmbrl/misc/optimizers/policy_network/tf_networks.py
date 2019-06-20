"""
    Basic cells of neural network, most of them from the legendary mentor
    Renjie Liao.
"""
import tensorflow as tf
import numpy as np

from . import tf_norm


def get_activation_func(activation_type):
    if activation_type == 'leaky_relu':
        activation_func = tf.nn.leaky_relu
    elif activation_type == 'tanh':
        activation_func = tf.nn.tanh
    elif activation_type == 'relu':
        activation_func = tf.nn.relu
    else:
        raise ValueError(
            "Unsupported activation type: {}!".format(activation_type)
        )
    return activation_func


def get_normalizer(normalizer_type, train=True):

    if normalizer_type == 'batch_norm':
        normalizer = tf_norm.batch_norm_with_train if train else \
            tf_norm.batch_norm_without_train

    elif normalizer_type == 'layer_norm':
        normalizer = tf_norm.layer_norm

    elif normalizer_type == 'none':
        normalizer = tf.identity

    else:
        raise ValueError(
            "Unsupported normalizer type: {}!".format(normalizer_type)
        )
    return normalizer


def normc_initializer(shape, seed=1234, stddev=1.0, dtype=tf.float32):
    npr = np.random.RandomState(seed)
    out = npr.randn(*shape).astype(np.float32)
    out *= stddev / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return tf.constant(out)


def normc_initializer_func(std=1.0, axis=0, seed=1234):
    def _initializer(shape, dtype=None, partition_info=None):
        npr = np.random.RandomState(seed)
        out = npr.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer


def weight_variable(shape, name, init_method=None, dtype=tf.float32,
                    init_para=None, seed=1234, trainable=True):
    """ @brief:
            Initialize weights

        @input:
            shape: list of int, shape of the weights
            init_method: string, indicates initialization method
            init_para: a dictionary,
            init_val: if it is not None, it should be a tensor

        @output:
            var: a TensorFlow Variable
    """

    if init_method is None or init_method == 'zero':
        initializer = tf.zeros_initializer(shape, dtype=dtype)

    if init_method == "normc":
        var = normc_initializer(
            shape, stddev=init_para['stddev'],
            seed=seed, dtype=dtype
        )
        return tf.get_variable(initializer=var, name=name, trainable=trainable)

    elif init_method == "normal":
        initializer = tf.random_normal_initializer(
            mean=init_para["mean"], stddev=init_para["stddev"],
            seed=seed, dtype=dtype
        )

    elif init_method == "truncated_normal":
        initializer = tf.truncated_normal_initializer(
            mean=init_para["mean"], stddev=init_para["stddev"],
            seed=seed, dtype=dtype
        )

    elif init_method == "uniform":
        initializer = tf.random_uniform_initializer(
            minval=init_para["minval"], maxval=init_para["maxval"],
            seed=seed, dtype=dtype
        )

    elif init_method == "constant":
        initializer = tf.constant_initializer(
            value=init_para["val"], dtype=dtype
        )

    elif init_method == "xavier":
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=init_para['uniform'], seed=seed, dtype=dtype
        )

    elif init_method == 'orthogonal':
        initializer = tf.orthogonal_initializer(
            gain=1.0, seed=seed, dtype=dtype
        )

    else:
        raise ValueError("Unsupported initialization method!")

    var = tf.get_variable(initializer=initializer(shape),
                          name=name, trainable=trainable)

    return var


class GRU(object):
    """ Gated Recurrent Units (GRU)

        Input:
                input_dim: input dimension
                hidden_dim: hidden dimension
                wd: a float, weight decay
                scope: tf scope of the model

        Output:
                a function which computes the output of GRU with one step
    """

    def __init__(self, input_dim, hidden_dim, init_method='truncated_normal',
                 dtype=tf.float32, init_std=None, scope="GRU"):

        self._init_method = init_method

        # initialize variables
        with tf.variable_scope(scope):
            self._w_xi = weight_variable(
                shape=[input_dim, hidden_dim], name="w_xi",
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std}, dtype=dtype
            )
            self._w_hi = weight_variable(
                shape=[hidden_dim, hidden_dim], name="w_hi",
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std}, dtype=dtype
            )
            self._b_i = weight_variable(
                shape=[hidden_dim], name="b_i",
                init_method="constant",
                init_para={"val": 0.0}, dtype=dtype
            )

            self._w_xr = weight_variable(
                shape=[input_dim, hidden_dim], name="w_xr",
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std}, dtype=dtype
            )
            self._w_hr = weight_variable(
                shape=[hidden_dim, hidden_dim], name="w_hr",
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std}, dtype=dtype
            )
            self._b_r = weight_variable(
                shape=[hidden_dim], name="b_r",
                init_method="constant",
                init_para={"val": 0.0}, dtype=dtype
            )

            self._w_xu = weight_variable(
                shape=[input_dim, hidden_dim], name="w_xu",
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std}, dtype=dtype
            )
            self._w_hu = weight_variable(
                shape=[hidden_dim, hidden_dim], name="w_hu",
                init_method=self._init_method,
                init_para={"mean": 0.0, "stddev": init_std}, dtype=dtype
            )
            self._b_u = weight_variable(
                shape=[hidden_dim], name="b_u",
                init_method="constant",
                init_para={"val": 0.0}, dtype=dtype
            )

    def __call__(self, x, state):
        # update gate
        g_i = tf.sigmoid(
            tf.matmul(x, self._w_xi) + tf.matmul(state, self._w_hi) + self._b_i
        )

        # reset gate
        g_r = tf.sigmoid(
            tf.matmul(x, self._w_xr) + tf.matmul(state, self._w_hr) + self._b_r
        )

        # new memory implementation 1
        u = tf.tanh(
            tf.matmul(x, self._w_xu) + tf.matmul(g_r * state, self._w_hu) +
            self._b_u
        )

        # hidden state
        new_state = state * g_i + u * (1 - g_i)

        return new_state


class MLP(object):
    """ Multi Layer Perceptron (MLP)
                Note: the number of layers is N

        Input:
                dims: a list of N+1 int, number of hidden units (last one is the
                input dimension)
                act_func: a list of N activation functions
                add_bias: a boolean, indicates whether adding bias or not
                scope: tf scope of the model

    """

    def __init__(self, dims, scope, train,
                 activation_type, normalizer_type, init_data,
                 dtype=tf.float32):

        self._scope = scope
        self._num_layer = len(dims) - 1  # the last one is the input dim
        self._w = [None] * self._num_layer
        self._b = [None] * self._num_layer
        self._train = train

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        self._init_data = init_data

        # initialize variables
        with tf.variable_scope(scope):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    dim_in, dim_out = dims[ii], dims[ii + 1]

                    self._w[ii] = weight_variable(
                        shape=[dim_in, dim_out], name='w',
                        init_method=self._init_data[ii]['w_init_method'],
                        init_para=self._init_data[ii]['w_init_para'],
                        dtype=dtype, trainable=self._train
                    )

                    self._b[ii] = weight_variable(
                        shape=[dim_out], name='b',
                        init_method=self._init_data[ii]['b_init_method'],
                        init_para=self._init_data[ii]['b_init_para'],
                        dtype=dtype, trainable=self._train
                    )

    def __call__(self, input_vec):
        self._h = [None] * self._num_layer

        with tf.variable_scope(self._scope):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    layer = input_vec if ii == 0 else self._h[ii - 1]
                    self._h[ii] = tf.matmul(layer, self._w[ii]) + self._b[ii]

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])
                        self._h[ii] = \
                            act_func(self._h[ii], name='activation_' + str(ii))

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizar_' + str(ii))

        return self._h[-1]

    def get_variable_list(self):
        return self._w + self._b


class W_MLP(MLP):
    """ Weight-input Multi Layer Perceptron (MLP)

        @brief: In this model, we can manipulate the weights directory
    """

    def __init__(self, dims, scope, train,
                 activation_type, normalizer_type, init_data,
                 dtype=tf.float32):

        self._scope = scope
        self._num_layer = len(dims) - 1  # the last one is the input dim
        self._w = [None] * self._num_layer
        self._b = [None] * self._num_layer
        self._train = train
        self._dims = dims

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        self._init_data = init_data

        self._w_flat_id, self._b_flat_id = [], []

        # initialize variables
        self.weight_size = 0
        with tf.variable_scope(scope):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    dim_in, dim_out = dims[ii], dims[ii + 1]

                    self._w[ii] = weight_variable(
                        shape=[dim_in, dim_out], name='w',
                        init_method=self._init_data[ii]['w_init_method'],
                        init_para=self._init_data[ii]['w_init_para'],
                        dtype=dtype, trainable=self._train,
                    )
                    self._w_flat_id.append(
                        [self.weight_size, self.weight_size + dim_in * dim_out]
                    )
                    self.weight_size += dim_in * dim_out

                    self._b[ii] = weight_variable(
                        shape=[dim_out], name='b',
                        init_method=self._init_data[ii]['b_init_method'],
                        init_para=self._init_data[ii]['b_init_para'],
                        dtype=dtype, trainable=self._train,
                    )
                    self._b_flat_id.append(
                        [self.weight_size, self.weight_size + dim_out]
                    )
                    self.weight_size += dim_out

    def get_weight_size(self):
        return self.weight_size

    def __call__(self, input_vec, weight_vec):
        self._h = [None] * self._num_layer
        input_vec = tf.reshape(input_vec, [-1, 1, self._dims[0]])

        w_noise, b_noise = self._parse_weight_vec(weight_vec)

        with tf.variable_scope(self._scope):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    layer = input_vec if ii == 0 else self._h[ii - 1]
                    self._h[ii] = tf.matmul(layer, self._w[ii] + w_noise[ii])
                    self._h[ii] = self._h[ii] + \
                        (self._b[ii] + b_noise[ii])

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])
                        self._h[ii] = \
                            act_func(self._h[ii], name='activation_' + str(ii))

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizar_' + str(ii))

        return tf.reshape(self._h[-1], [-1, self._dims[-1]])

    def _parse_weight_vec(self, weight_vec):
        w_noise, b_noise = [], []
        for ii in range(self._num_layer):
            dim_in, dim_out = self._dims[ii], self._dims[ii + 1]

            # the w weight noise
            noise_weight = tf.reshape(
                weight_vec[:, self._w_flat_id[ii][0]: self._w_flat_id[ii][1]],
                [-1, dim_in, dim_out]
            )
            w_noise.append(noise_weight)

            # the b weight noise
            noise_weight = tf.reshape(
                weight_vec[:, self._b_flat_id[ii][0]: self._b_flat_id[ii][1]],
                [-1, 1, dim_out]
            )
            b_noise.append(noise_weight)
        return w_noise, b_noise

    def parse_np_weight_vec_into_dict(self, weight_vec):
        data_dict = {}
        for ii in range(self._num_layer):
            dim_in, dim_out = self._dims[ii], self._dims[ii + 1]

            var_weight = np.reshape(
                weight_vec[:, self._w_flat_id[ii][0]: self._w_flat_id[ii][1]],
                [dim_in, dim_out]
            )
            data_dict[self._w[ii].name] = var_weight

            var_weight = np.reshape(
                weight_vec[:, self._b_flat_id[ii][0]: self._b_flat_id[ii][1]],
                [dim_out]
            )
            data_dict[self._b[ii].name] = var_weight
        return data_dict


def flatten_feature(x):
    return tf.reshape(x, [-1, int(np.prod(x.get_shape().as_list()[1:]))])


def conv2d(x, num_filters, name, filter_size, stride,
           w_variable_list, b_variable_list, pad="same", dtype=tf.float32):

    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1],
                        int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = int(np.prod(filter_shape[:3]))
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = int(np.prod(filter_shape[:2])) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("w", filter_shape, dtype,
                            tf.random_uniform_initializer(-w_bound, w_bound))
        b = tf.get_variable("b", [1, 1, 1, num_filters],
                            initializer=tf.zeros_initializer())

        w_variable_list.append(w)
        b_variable_list.append(b)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


class CNN_RL(object):
    """ Cnn for policy network. we allow for the rl policy network.
        the one used in dqn, and the one used in a2c
    """

    def __init__(self, output_size, scope, cnn_type, train,
                 activation_type, normalizer_type,
                 dtype=tf.float32):

        self._scope = scope
        self._train = train
        self._cnn_type = cnn_type
        self._output_size = output_size

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        assert self._activation_type == 'relu'
        assert self._normalizer_type is None

        self._w = []
        self._b = []

    def __call__(self, input_vec):
        self._input = input_vec / 255.0
        self._h = []

        with tf.variable_scope(self._scope):

            if self._cnn_type == 'a3c':  # from A3C paper

                # first layer
                layer = conv2d(self._input, 16, "l1", [8, 8], [4, 4],
                               self._w, self._b, pad="VALID")
                self._h.append(tf.nn.relu(layer))

                # second layer
                layer = conv2d(self._h[-1], 32, "l2", [4, 4], [2, 2],
                               self._w, self._b, pad="VALID")
                self._h.append(tf.nn.relu(layer))

                # the dense layer
                layer = tf.layers.dense(
                    flatten_feature(self._h[-1]), 256,
                    name='lin', kernel_initializer=normc_initializer_func(1.0)
                )
                self._h.append(tf.nn.relu(layer))

            elif self._cnn_type == 'dqn':  # Nature DQN

                # first layer
                layer = conv2d(self._input, 32, "l1", [8, 8], [4, 4],
                               self._w, self._b, pad="VALID")
                self._h.append(tf.nn.relu(layer))

                # second layer
                layer = conv2d(self._h[-1], 64, "l2", [4, 4], [2, 2],
                               self._w, self._b, pad="VALID")
                self._h.append(tf.nn.relu(layer))

                # third layer
                layer = conv2d(self._h[-1], 64, "l3", [3, 3], [1, 1],
                               self._w, self._b, pad="VALID")
                self._h.append(tf.nn.relu(layer))

                # the dense layer
                layer = tf.layers.dense(
                    flatten_feature(self._h[-1]), 512,
                    name='lin', kernel_initializer=normc_initializer_func(1.0)
                )
                self._h.append(tf.nn.relu(layer))

            else:
                raise NotImplementedError

            self._logits = tf.layers.dense(
                self._h[-1], self._output_size,
                name='logits', kernel_initializer=normc_initializer_func(0.01)
            )
            # self.pd = pdtype.pdfromflat(logits)
            self._vpred = tf.layers.dense(
                self._h[-1], 1, name='value',
                kernel_initializer=normc_initializer_func(1.0)
            )[:, 0]

            raise NotImplementedError

            '''
            stochastic = tf.placeholder(dtype=tf.bool, shape=())
            ac = self.pd.sample()  # XXX
            self._act = U.function([stochastic, ob], [ac, self.vpred])
            '''

        return self._h[-1]


class WZ_MLP(W_MLP):
    """ Weight-input Multi Layer Perceptron (MLP)

        @brief: In this model, we can manipulate the weights directory
    """

    def __init__(self, dims, scope, train,
                 activation_type, normalizer_type, init_data,
                 dtype=tf.float32):

        self._scope = scope
        self._num_layer = len(dims) - 1  # the last one is the input dim
        self._w = [None] * self._num_layer
        self._b = [None] * self._num_layer
        self._train = train
        self._dims = dims

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        self._init_data = init_data

        self._w_flat_id, self._b_flat_id = [], []

        # initialize variables
        self.weight_size = 0
        with tf.variable_scope(scope):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    dim_in, dim_out = dims[ii], dims[ii + 1]

                    self._w_flat_id.append(
                        [self.weight_size, self.weight_size + dim_in * dim_out]
                    )
                    self.weight_size += dim_in * dim_out

                    self._b_flat_id.append(
                        [self.weight_size, self.weight_size + dim_out]
                    )
                    self.weight_size += dim_out

    def __call__(self, input_vec, weight_vec):
        self._h = [None] * self._num_layer
        input_vec = tf.reshape(input_vec, [-1, 1, self._dims[0]])

        w_noise, b_noise = self._parse_weight_vec(weight_vec)

        with tf.variable_scope(self._scope):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    layer = input_vec if ii == 0 else self._h[ii - 1]
                    # it's different
                    self._h[ii] = tf.matmul(layer, w_noise[ii])
                    self._h[ii] = self._h[ii] + b_noise[ii]

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])
                        self._h[ii] = \
                            act_func(self._h[ii], name='activation_' + str(ii))

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizar_' + str(ii))

        return tf.reshape(self._h[-1], [-1, self._dims[-1]])
