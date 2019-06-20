# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf

from . import whitening_util
from . import tf_utils
from dmbrl.misc import logger


def limit_action(action, lb=-1, ub=1):

    return tf.minimum(tf.maximum(action, lb), ub)


class base_policy_network(object):
    '''
        @brief:
            In this object class, we define the network structure, the restore
            function and save function.
            It will only be called in the agent/agent.py
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):
        self.args = args

        self._session = session
        self._name_scope = name_scope

        self._observation_size = observation_size
        self._action_size = action_size

        # self._task_name = args.task_name
        self._network_shape = args.policy_network_shape

        self._npr = np.random.RandomState(args.seed)

        self._whitening_operator = {}
        self._whitening_variable = []

    def build_network(self):
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError

    def _build_ph(self):

        # initialize the running mean and std (whitening)
        whitening_util.add_whitening_operator(
            self._whitening_operator, self._whitening_variable,
            'state', self._observation_size
        )

        # initialize the input placeholder
        self._input_ph = {
            'start_state': tf.placeholder(
                tf.float32, [None, self._observation_size], name='start_state'
            )
        }

    def get_input_placeholder(self):
        return self._input_ph

    def get_weights(self):
        return None

    def set_weights(self, weights_dict):
        pass

    def forward_network(self, observation, weight_vec=None):
        normalized_start_state = (
            observation - self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']

        # the output policy of the network
        if weight_vec is None:
            action = self._MLP(normalized_start_state)
        else:
            action = self._MLP(normalized_start_state, weight_vec)

        action = limit_action(action)

        return action

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]

        # the weights that actually matter
        self._network_var_list = \
            self._trainable_var_list + self._whitening_variable

        self._set_network_weights = tf_utils.set_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

        self._get_network_weights = tf_utils.get_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

    def load_checkpoint(self, ckpt_path):
        pass

    def save_checkpoint(self, ckpt_path):
        pass

    def get_whitening_operator(self):
        return self._whitening_operator

    def _set_whitening_var(self, whitening_stats):
        whitening_util.set_whitening_var(
            self._session, self._whitening_operator, whitening_stats, ['state']
        )

    def train(self, data_dict, replay_buffer, training_info={}):
        raise NotImplementedError

    def eval(self, data_dict):
        raise NotImplementedError

    def act(self, data_dict):
        raise NotImplementedError

    def optimize_weights(self, data_dict, training_keys):

        test_set_id = np.arange(len(data_dict['start_state']))
        num_test_data = int(len(test_set_id) * self.args.pct_testset)
        self._npr.shuffle(test_set_id)
        test_set = {key: data_dict[key][test_set_id][:num_test_data]
                    for key in training_keys}
        train_set = {key: data_dict[key][test_set_id][num_test_data:]
                     for key in training_keys}
        test_error = old_test_error = np.inf

        # supervised training the behavior (behavior cloning)
        for epoch in range(self.args.policy_epochs):
            total_batch_len = len(train_set['start_state'])
            total_batch_inds = np.arange(total_batch_len)
            self._npr.shuffle(total_batch_inds)
            num_minibatch = \
                max(total_batch_len // self.args.minibatch_size, 1)
            train_error = []

            for start in range(num_minibatch):
                start = start * self.args.minibatch_size
                end = min(start + self.args.minibatch_size, total_batch_len)
                batch_inds = total_batch_inds[start: end]
                feed_dict = {self._input_ph[key]: data_dict[key][batch_inds]
                             for key in training_keys}

                error, _ = self._session.run(
                    [self._update_operator['loss'],
                     self._update_operator['update_op']], feed_dict=feed_dict
                )
                train_error.append(error)

            # see the test error
            feed_dict = {self._input_ph[key]: test_set[key]
                         for key in training_keys}

            test_error = self._session.run(
                self._update_operator['loss'], feed_dict=feed_dict
            )
            logger.info('Epoch %d; Train Error: %.6f; Test Error: %.6f' %
                        (epoch, np.mean(train_error), test_error))

            if test_error > old_test_error and epoch % 5 == 0:
                # TODO: MAKE A COUNTER HERE
                logger.info('Early stoping')
                break
            else:
                old_test_error = test_error
