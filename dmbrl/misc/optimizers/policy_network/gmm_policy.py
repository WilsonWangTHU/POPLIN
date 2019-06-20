# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

from . import base_policy
from . import tf_networks
from . import gmm_util
# from . import tf_utils
# from dmbrl.misc import logger
from sklearn.mixture import GaussianMixture


class policy_network(base_policy.base_policy_network):
    ''' @brief:
            In this object class, we define the network structure, the restore
            function and save function.

        @self.args.training_scheme
            @G-R: using real dataset, model a gaussian distribution (like a
                outer loop CEM)
            @G-I: using imaginary dataset

            @GMM-R: using real dataset. Model a conditional GMM on top of the
                dataset. Conditioned on the observation, model the weights
            @GMM-I: using the imaginary_dataset
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):

        super(policy_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        # In total, we have 4 PGMMWCEM methods and 2 PGWCEM methods

        if self.args.cem_type in ['PGWCEM-TRAJ', 'PGWCEM-ALL']:
            # TRAJ means sharing the weight, ALL means different possibilities
            assert self.args.training_scheme in ['G-R', 'G-I']

        elif self.args.cem_type in ['PGMMWCEM-TRAJ']:
            # PGMMWCEM-TRAJ means a single distribution ()
            assert self.args.training_scheme in ['GMM-R']

        elif self.args.cem_type in ['PGMMWCEM-ALLSP']:
            assert self.args.training_scheme in ['GMM-R', 'GMM-I']

        elif self.args.cem_type in ['PGMMWCEM-ALLM']:
            assert self.args.training_scheme in ['GMM-I']

        elif self.args.cem_type in ['PGMMWCEM-ALLR']:
            assert self.args.training_scheme in ['GMM-R', 'GMM-I']

    def forward_network(self, observation, weight_vec=None):
        """ @brief: similar to the bc_policy
        """
        if self.args.cem_type not in ['PGMMWCEM-ALLR']:
            raise NotImplementedError
        else:
            normalized_start_state = (
                observation - self._whitening_operator['state_mean']
            ) / self._whitening_operator['state_std']

            # the output policy of the network
            if weight_vec is None:
                action = self._MLP(normalized_start_state)
            else:
                action = self._MLP(normalized_start_state, weight_vec)

            action = base_policy.limit_action(action)

        return action

    def build_network(self):
        """ @brief: Note that build_network is only needed for the training
        """

        # the policy network
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

        if self.args.cem_type in ['PGMMWCEM-ALLR']:
            raise NotImplementedError
        elif self.args.cem_type in ['PGWCEM-TRAJ', 'PGWCEM-ALL',
                                    'PGMMWCEM-TRAJ', 'PGMMWCEM-ALLSP',
                                    'PGMMWCEM-ALLM']:
            self._MLP = tf_networks.WZ_MLP(
                dims=network_shape, scope='policy_mlp', train=True,
                activation_type=act_type, normalizer_type=norm_type,
                init_data=init_data
            )
        else:
            raise NotImplementedError

        # step 2: build the gmm models
        if self.args.cem_type in ['PGMMWCEM-ALLR']:
            raise NotImplementedError
        elif self.args.cem_type in ['PGWCEM-TRAJ', 'PGWCEM-ALL']:
            self._gmm = GaussianMixture(
                1, covariance_type='full',
                max_iter=self.args.gmm_max_iteration,
                random_state=self.args.seed, warm_start=True
            )
            self._gmm_weights = {'mean': None, 'cov': None}
        elif self.args.cem_type in ['PGMMWCEM-TRAJ', 'PGMMWCEM-ALLSP']:
            self._gmm = GaussianMixture(
                self.args.gmm_num_cluster, covariance_type='full',
                max_iter=self.args.gmm_max_iteration,
                random_state=self.args.seed, warm_start=True
            )
            self._gmm_weights = {'mean': None, 'cov': None}
        elif self.args.cem_type in ['PGMMWCEM-ALLM']:
            self._gmm, self._gmm_weights = [], []
            for _ in range(self.args.plan_hor):
                i_gmm = GaussianMixture(
                    self.args.gmm_num_cluster, covariance_type='full',
                    max_iter=self.args.gmm_max_iteration,
                    random_state=self.args.seed, warm_start=True
                )
                i_gmm_weights = {'mean': None, 'cov': None}
                self._gmm.append(i_gmm)
                self._gmm_weights.append(i_gmm_weights)
        else:
            raise NotImplementedError

        self._set_var_list()

    def build_loss(self):
        self._session.run(tf.variables_initializer(tf.global_variables()))

        # build the GMM / Gaussian models

    def train(self, data_dict, training_info={}):
        """ @brief:
                @data_dict:
                    when passed into the train function, data_dict is a
                    dictionary of 'weight', 'start_state'

                    Before it's able to be used by training, it needs to be
                    preprocessed so that:
                    for G-R / G-I: a dictionary of 'start_state', 'weight'
                    for GMM-R: a dictionary of 'start_state', 'weight'
                    for GMM-I: a dictionary of 'start_state', 'weight_0', ...
                        'weight_n'
                @training_info['imaginary_dataset']
                    A dictionary of 'plan_weight' (different shape, plan_hor *
                    weight_size), and 'start_state'
        """

        # Step 1: update the running mean
        imaginary_dataset = training_info['imaginary_dataset']

        # Step 2: data processing
        if self.args.training_scheme in ['G-R', 'GMM-R']:
            assert 'weight' in data_dict and 'start_state' in data_dict
            assert len(data_dict['weight']) == len(data_dict['start_state'])

        elif self.args.training_scheme in ['G-I', 'GMM-I']:
            # add imaginary data to the dataset
            start_state, weight = [data_dict['start_state']], [data_dict['weight']]
            data_dict['real_start_state'] = data_dict['start_state']
            data_dict['weight_0'] = data_dict['weight']
            for i_depth in range(1, self.args.plan_hor):
                start_state.append(imaginary_dataset['start_state'])
                weight.append(imaginary_dataset['plan_weight'][:, i_depth])
                data_dict['weight_' + str(i_depth)] = \
                    imaginary_dataset['plan_weight'][:, i_depth]

            data_dict['start_state'] = np.concatenate(start_state)
            data_dict['weight'] = np.concatenate(weight)

        else:
            raise NotImplementedError

        # Step 3: train the network
        if self.args.cem_type in ['PGWCEM-TRAJ', 'PGWCEM-ALL']:
            self.fit_gaussian_distribution(data_dict)

        elif self.args.training_scheme in ['PGMMWCEM-TRAJ', 'PGMMWCEM-ALLSP',
                                           'PGMMWCEM-ALLR']:
            self.fit_gmm_distribution(data_dict)

        elif self.args.training_scheme in ['PGMMWCEM-ALLM']:
            self.fit_multiple_gmm_distribution(data_dict)

        else:
            raise NotImplementedError

    def fit_gaussian_distribution(self, data_dict):
        """ @brief: it is using the GMM to do the guassian distribution fit
            self._gmm = GaussianMixture(
                1, covariance_type='full',
                max_iter=self.args.gmm_max_iteration,
                random_state=self.args.seed, warm_start=True
            )
            self._gmm_weights = {'mean': None, 'cov': None}
        """
        self._size_info = {'ob_size': data_dict['start_state'].shape[1],
                           'weight_size': data_dict['weight'].shape[1]}
        train_data = np.concatenate(data_dict['weight'])
        self._gmm.fit(train_data)

        self._gmm_weights['mean'], self._gmm_weights['cov'] = \
            self._gmm.means_, self._gmm.covariances_

    def fit_gmm_distribution(self, data_dict):
        train_data = np.concatenate([data_dict['start_state'],
                                     data_dict['weight']], axis=1)
        self._gmm.fit(train_data)

        self._gmm_weights['mean'] = self._gmm.means_
        self._gmm_weights['cov'] = self._gmm.covariances_

    def fit_multiple_gmm_distribution(self, data_dict):
        for i_gmm in range(self.args.plan_hor):
            train_data = np.concatenate(
                [data_dict['start_state'], data_dict['weight_' + str(i_gmm)]],
                axis=1
            )

            self._gmm[i_gmm].fit(train_data)
            self._gmm_weights[i_gmm]['mean'] = self._gmm[i_gmm].means_
            self._gmm_weights[i_gmm]['cov'] = self._gmm[i_gmm].covariances_

    def get_posterior_distribution(self, init_ob, timestep=0):
        gmm_weights = self._gmm_weights if type(self._gmm_weights) is list \
            else self._gmm_weights[timestep]
        if self.args.training_scheme in ['G-I', 'G-R']:
            return gmm_weights['mean'], gmm_weights['cov']

        elif self.args.training_scheme in ['GMM-I', 'GMM-R']:
            # get the conditioanl guassian distribution
            if self.args.cem_type == 'PGMMWCEM-ALLM':
                conditioned_mean, conditioned_cov = [], []
                for i_step in range(self.args.plan_hor):
                    posterior_mean, posterior_cov = gmm_util.get_gmm_posterior(
                        self._gmm[i_step], self._gmm_weights, init_ob
                    )
                    i_conditioned_gauss = gmm_util.get_conditional_gaussian(
                        posterior_mean, posterior_cov,
                        self._size_info['ob_size']
                    )
                    i_conditioned_mean = i_conditioned_gauss['f_c'] + \
                        i_conditioned_gauss['f_d'].dot(init_ob)
                    i_conditioned_cov = i_conditioned_gauss['cov']

                    conditioned_mean.append(i_conditioned_mean)
                    conditioned_cov.append(i_conditioned_cov)
                return conditioned_mean, conditioned_cov
            else:
                posterior_mean, posterior_cov = gmm_util.get_gmm_posterior(
                    self._gmm[i_step], self._gmm_weights, init_ob
                )
                conditioned_gauss = gmm_util.get_conditional_gaussian(
                    posterior_mean, posterior_cov, self._size_info['ob_size']
                )
                conditioned_mean = conditioned_gauss['f_c'] + \
                    conditioned_gauss['f_d'].dot(init_ob)
                conditioned_cov = conditioned_gauss['cov']

                return conditioned_mean, conditioned_cov
        else:
            assert False
