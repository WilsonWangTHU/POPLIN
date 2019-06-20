# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

from . import base_policy
from . import tf_networks
from dmbrl.misc import logger


class policy_network(base_policy.base_policy_network):
    ''' @brief:
            In this object class, we define the network structure, the restore
            function and save function.

        @self.args.training_scheme
            @BC-AR: (action space) behavior cloning with the real data
            @BC-AI: (action space) behavior cloning using imaginary dataset.

            @AVG-R: (weight space) behavior cloning by setting the weight to
                the average of the weights selected during sampling
            @BC-PR: (weight space) behavior cloning by distilling the policy
                produced by the weights during sampling
            @AVG-I: (weight space) AVG-R but with imaginary dataset
            @BC-PI: (weight space) BC-PR but with imaginary dataset
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):

        super(policy_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        assert self.args.training_scheme in ['BC-AR', 'BC-AI']
        assert self.args.cem_type in ['POPLINA-INIT', 'POPLINA-REPLAN']

    def build_network(self):
        """ @brief: Note that build_network is only needed for the training
        """
        network_shape = [self._observation_size] + \
            self.args.policy_network_shape + [self._action_size]
        num_layer = len(network_shape) - 1
        act_type = ['tanh'] * (num_layer - 1) + [None]
        norm_type = [None] * (num_layer - 1) + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        init_data[-1]['w_init_para']['stddev'] = 0.01  # the output layer std

        self._MLP = tf_networks.MLP(
            dims=network_shape, scope='policy_mlp', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )

        # fetch all the trainable variables
        self._set_var_list()

    def build_loss(self):

        self._build_ph()
        self._tensor, self._update_operator = {}, {}

        # construct the input to the forward network, we normalize the state
        # input, and concatenate with the action
        self._tensor['normalized_start_state'] = (
            self._input_ph['start_state'] -
            self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']
        self._tensor['net_input'] = self._tensor['normalized_start_state']

        # the output policy of the network
        self._tensor['action'] = self._MLP(self._tensor['net_input'])

        self._input_ph['target_action'] = tf.placeholder(
            tf.float32, [None, self._action_size], name='target_action'
        )

        self._update_operator['loss'] = tf.reduce_mean(
            tf.square(self._input_ph['target_action'] -
                      self._tensor['action'])
        )

        self._update_operator['update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.policy_lr,
        ).minimize(self._update_operator['loss'])
        logger.info("policy training learning rate: {}".format(
            self.args.policy_lr)
        )

    def train(self, data_dict, training_info={}):

        # Step 1: update the running mean
        imaginary_dataset = training_info['imaginary_dataset']

        # Step 2: data processing
        if self.args.training_scheme == 'BC-AR':
            data_dict['target_action'] = data_dict['action']  # for training
        elif self.args.training_scheme == 'BC-AI':
            # add imaginary data to the dataset
            for key in ['start_state', 'action']:
                data_dict[key] = \
                    np.concatenate([data_dict[key], imaginary_dataset[key]])
            data_dict['target_action'] = data_dict['action']  # for training

        else:
            raise NotImplementedError

        self._set_whitening_var(data_dict['whitening_stats'])
        self.optimize_weights(data_dict, ['start_state', 'target_action'])
