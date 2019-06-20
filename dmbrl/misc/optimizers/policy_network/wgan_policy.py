# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

from . import base_policy
from . import tf_networks
from . import tf_utils
from . import whitening_util
from dmbrl.misc import logger
from scipy import stats


def generate_noise(template_array, init_var):
    noise = stats.truncnorm(
        -2, 2, loc=np.zeros_like(template_array),
        scale=np.ones_like(template_array)
    )
    noise = noise.rvs(
        size=[template_array.shape[0], template_array.shape[1]]
    ) * np.sqrt(init_var)
    return noise


class policy_network(base_policy.base_policy_network):
    ''' @brief:
            In this object class, we define the network structure, the restore
            function and save function.

        @self.args.training_scheme
            @GAN-A: (weight space) AVG-R but with imaginary dataset
            @GAN-R: (weight space) BC-PR but with imaginary dataset
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):

        super(policy_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        assert self.args.cem_type in ['POPLINP-SEP', 'POPLINP-UNI']
        assert self.args.training_scheme in ['GAN-I', 'GAN-R']
        assert self.args.gan_type == 'WGAN'

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

        self._MLP = tf_networks.W_MLP(
            dims=network_shape, scope='policy_mlp', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )
        self._target_MLP = tf_networks.W_MLP(
            dims=network_shape, scope='target_policy_mlp', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )

        # fetch all the trainable variables
        self._set_var_list()

    def _build_policy_train_graph(self):
        """ @brief: build the computation graph related to the policy network
            for training
        """
        # the weight placeholder to be used during training
        self._input_ph['weight'] = tf.placeholder(
            shape=[None, self._MLP.get_weight_size()],
            dtype=tf.float32, name='weight_noise'
        )
        self._input_ph['target_weight'] = tf.placeholder(
            shape=[None, self._MLP.get_weight_size()], dtype=tf.float32,
            name='target_weight_noise'
        )
        whitening_util.add_whitening_operator(
            self._whitening_operator, self._whitening_variable,
            'target_state', self._observation_size
        )
        self._tensor['net_input'] = (
            self._input_ph['start_state'] -
            self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']
        self._tensor['target_net_input'] = (
            self._input_ph['start_state'] -
            self._whitening_operator['target_state_mean']
        ) / self._whitening_operator['target_state_std']

        # the output policy of the network
        self._tensor['action'] = self._MLP(
            self._tensor['net_input'], self._input_ph['weight']
        )
        self._tensor['target_action'] = self._target_MLP(
            self._tensor['target_net_input'],
            self._input_ph['target_weight']
        )

        self._target_MLP_var_list = self._target_MLP.get_variable_list()
        self._MLP_var_list = self._MLP.get_variable_list()

    def _build_discriminator(self):
        """ @brief: build the discriminator network
        """
        # step 1: build the MLP for discrimination
        network_shape = [self._observation_size + self._action_size] + \
            self.args.discriminator_network_shape + [1]
        num_layer = len(network_shape) - 1
        act_type = [self.args.discriminator_act_type] * (num_layer - 1) + [None]
        norm_type = \
            [self.args.discriminator_norm_type] * (num_layer - 1) + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )

        self._discriminator = tf_networks.MLP(
            dims=network_shape, scope='discriminator', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )
        self._discriminator_var_list = self._discriminator.get_variable_list()

        # step 2: process the logits
        with tf.variable_scope('disc_network'):
            self._tensor['target_disc_logits'] = self._discriminator(
                tf.concat([self._tensor['target_action'],
                           self._tensor['net_input']], axis=1)
            )
        with tf.variable_scope('disc_network', reuse=tf.AUTO_REUSE):
            self._tensor['disc_logits'] = self._discriminator(
                tf.concat([self._tensor['action'],
                           self._tensor['net_input']], axis=1)
            )

    def build_loss(self):

        self._build_ph()
        self._tensor, self._update_operator = {}, {}
        self._build_policy_train_graph()
        self._build_discriminator()

        self._update_operator['true_score'] = \
            tf.reduce_mean(self._tensor['target_disc_logits'])
        self._update_operator['fake_score'] = \
            tf.reduce_mean(self._tensor['disc_logits'])
        self._update_operator['distance_loss'] = \
            self._update_operator['fake_score'] - \
            self._update_operator['true_score']

        # the gradient penalty
        self._input_ph['alpha_interp'] = tf.random_uniform(
            shape=[self.args.discriminator_minibatch_size, 1],
            minval=0., maxval=1., name='alpha'
        )

        self._tensor['interp_action'] = self._tensor['target_action'] + \
            self._input_ph['alpha_interp'] * \
            (self._tensor['action'] - self._tensor['target_action'])

        with tf.variable_scope('disc_network', reuse=tf.AUTO_REUSE):
            self._tensor['interp_disc_logits'] = self._discriminator(
                tf.concat([self._tensor['interp_action'],
                           self._tensor['net_input']], axis=1)
            )

        self._tensor['gradient'] = tf.concat(
            tf.gradients(
                self._tensor['interp_disc_logits'],
                [self._tensor['interp_action'], self._tensor['net_input']]
            ), axis=1
        )
        self._tensor['gradient_norm'] = \
            tf.sqrt(tf.reduce_sum(tf.square(self._tensor['gradient']), axis=1))
        self._update_operator['gradient_penalty_loss'] = tf.reduce_mean(
            tf.square(self._tensor['gradient_norm'] - 1),
            name='gradient_penalty'
        ) * self.args.discriminator_gradient_penalty_coeff
        self._update_operator['gradient_norm_mean'] = \
            tf.reduce_mean(self._tensor['gradient_norm'])

        self._update_operator['discriminator_loss'] = \
            self._update_operator['distance_loss'] + \
            self._update_operator['gradient_penalty_loss']

        # self.g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')
        # self._tensor['disc_logits'] - self._tensor['target_disc_logits']
        self._update_operator['policy_loss'] = \
            -tf.reduce_mean(self._tensor['disc_logits'])

        self._update_operator['disc_update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.discriminator_lr,
        ).minimize(self._update_operator['discriminator_loss'],
                   var_list=self._discriminator_var_list)
        self._update_operator['policy_update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.policy_lr,
        ).minimize(self._update_operator['policy_loss'],
                   var_list=self._MLP_var_list)

        # synchronize the weights
        self._get_weight = tf_utils.get_network_weights(
            self._session, self._MLP_var_list, 'policy_mlp'
        )
        self._set_weight = tf_utils.set_network_weights(
            self._session, self._target_MLP_var_list, 'target_policy_mlp'
        )

        self._session.run(tf.variables_initializer(tf.global_variables()))
        self._set_weight(self._get_weight())

    def train(self, data_dict, training_info={}):

        # step 1: update the running mean
        imaginary_dataset = training_info['imaginary_dataset']

        # step 2: data processing
        if self.args.training_scheme in ['GAN-R']:
            data_dict['target_weight'] = data_dict['weight']  # for training

        elif self.args.training_scheme in ['GAN-I']:
            for key in ['start_state', 'weight']:
                data_dict[key] = \
                    np.concatenate([data_dict[key], imaginary_dataset[key]])
            data_dict['target_weight'] = data_dict['weight']  # for training

        else:
            raise NotImplementedError

        # step 3: parse the test set and train the network
        self.optimize_weights(data_dict,
                              ['start_state', 'target_weight', 'weight'])

    def optimize_weights(self, data_dict, training_keys):
        self._set_whitening_var(data_dict['whitening_stats'])

        for i_epoch in range(self.args.discriminator_epochs):
            # step 1: generate the GAN noise, the training_ids
            data_dict['weight'] = \
                generate_noise(data_dict['target_weight'], self.args.init_var)

            data_id = np.arange(len(data_dict['start_state']))
            self._npr.shuffle(data_id)
            num_minibatch = max(len(data_id) //
                                self.args.discriminator_minibatch_size, 1)
            recorder = {'disc_loss': [], 'policy_loss': [],
                        'gradient_norm': [], 'gradient_loss': [],
                        'fake_score': [], 'true_score': []}

            for start in range(num_minibatch):
                start_id = start * self.args.discriminator_minibatch_size
                end_id = min(start_id + self.args.discriminator_minibatch_size,
                             len(data_dict['start_state']))
                batch_inds = data_id[start_id: end_id]
                feed_dict = {self._input_ph[key]: data_dict[key][batch_inds]
                             for key in training_keys}
                # step 2: optimize the discriminator
                disc_log = self._session.run(
                    {'disc_loss': self._update_operator['discriminator_loss'],
                     'gradient_norm': self._update_operator['gradient_norm_mean'],
                     'gradient_loss': self._update_operator['gradient_penalty_loss'],
                     'op': self._update_operator['disc_update_op'],
                     'true_score': self._update_operator['true_score']},
                    feed_dict=feed_dict
                )

                # step 3: optimize the generator (train the policy network)
                policy_log = self._session.run(
                    {'policy_loss': self._update_operator['policy_loss'],
                     'op': self._update_operator['policy_update_op'],
                     'fake_score': self._update_operator['fake_score']},
                    feed_dict=feed_dict
                )

                policy_log.update(disc_log)
                for key in recorder:
                    recorder[key].append(policy_log[key])

            logger.info("GAN policy epoch: {}".format(i_epoch))
            for key in recorder:
                logger.info("\t[loss] " + key + ": " + "%.6f" % np.mean(recorder[key]))

        # step 4: synchronize the target network
        self._set_weight(self._get_weight())
        whitening_util.copy_whitening_var(data_dict['whitening_stats'],
                                          'state', 'target_state')
        whitening_util.set_whitening_var(
            self._session, self._whitening_operator,
            data_dict['whitening_stats'], ['target_state']
        )
