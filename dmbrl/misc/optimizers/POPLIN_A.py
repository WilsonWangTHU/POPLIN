from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
# from dmbrl.misc.DotmapUtils import get_required_argument

# import scipy.stats as stats

from .optimizer import Optimizer
# from .policy_network import bc_policy
from .policy_network import BC_A_policy
from .policy_network import BC_WD_policy
from .policy_network import BC_WA_policy
from .policy_network import gan_policy
from .policy_network import wgan_policy
from .policy_network import whitening_util


class POPLINAOptimizer(Optimizer):
    """A Tensorflow-compatible CEM optimizer.

        In CEM, we use a population based search algorithm (evolutionary search)
        This might be quite local, as it might be overfitting, and hard to find
        policy for a complex structure like humanoid.

        We use a policy network to choose the action.

        1. CEM strategy and Noise strategy:
            @POPLINA-INIT: Adding noise in the action space. Using a policy network as
                the initial proposal, and using CEM to get fine-grained action.
            @POPLINA-REPLAN: Adding noise in the action space. Using a policy network as
                the initial proposal, and using CEM to get fine-grained action.
            @POPLINP-SEP: Adding noise in the weight space. Using a PW-CEM policy.
                For each output, we have separate noise
            @POPLINP-UNI: Adding noise in the weight space. Using a PW-CEM
                policy for each candaites, with different CEM noise.
            @PACEM: noise in the activation space (this might not be useful)

        2. training_scheme
            @BC-AR: behavior cloning training only with the real data
            @BC-AI: behavior cloning training, train towards the action (added
                by the noise) during planning (imaginary_dataset).
            TODO

            @PPO-R: standard PPO / TRPO training
            @PPO-AH: standard PPO / TRPO training

            @SAC: the soft-actor-critic? This could be quite sample efficient
                @SAC-R (real), @SAC-AH
    """

    def __init__(self, sol_dim, max_iters, popsize, num_elites, tf_session=None,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25,
                 params=None):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            tf_session (tf.Session): (optional) Session to be used for this optimizer. Defaults to None,
                in which case any functions passed in cannot be tf.Tensor-valued.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.max_iters, self.popsize, self.num_elites = \
            max_iters, popsize, num_elites
        self._params = params
        self._cem_type = params.cem_cfg.cem_type
        self.set_sol_dim(sol_dim)
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha
        self.tf_sess = tf_session
        self._dataset = []

        self._whitening_stats = whitening_util.init_whitening_stats(['state'])

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        if self.tf_sess is not None:
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("CEMSolver"):
                    self.init_mean = \
                        tf.placeholder(dtype=tf.float32, shape=[self.sol_dim])
                    self.init_var = \
                        tf.placeholder(dtype=tf.float32, shape=[self.sol_dim])

        self.num_opt_iters, self.mean, self.var = None, None, None
        self.tf_compatible, self.cost_function = None, None

        # behavior cloning network
        with self.tf_sess.graph.as_default():
            with tf.variable_scope("bc_policy_network"):
                if self._params.cem_cfg.training_scheme in ['BC-AR', 'BC-AI']:
                    self._policy_network = BC_A_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_bc_network',
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0]
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                elif self._params.cem_cfg.training_scheme in ['BC-PI', 'BC-PR']:
                    self._policy_network = BC_WD_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_bc_network',
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0]
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                elif self._params.cem_cfg.training_scheme in ['AVG-I', 'AVG-R']:
                    self._policy_network = BC_WA_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_bc_network',
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0]
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                elif self._params.cem_cfg.gan_type == 'GAN' and \
                        self._params.cem_cfg.training_scheme in \
                        ['GAN-R', 'GAN-I']:
                    self._policy_network = gan_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_bc_network',
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0]
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                elif self._params.cem_cfg.gan_type == 'WGAN' and \
                        self._params.cem_cfg.training_scheme in \
                        ['GAN-R', 'GAN-I']:
                    self._policy_network = wgan_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_bc_network',
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0]
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                else:
                    raise NotImplementedError

    def set_sol_dim(self, sol_dim):

        self.dO, self.dU = self._params.env.observation_space.shape[0], \
            self._params.env.action_space.shape[0]
        self.plan_hor = self._params.opt_cfg.plan_hor
        self.npart = self._params.prop_cfg.npart  # number of dynamics particles

        if self._params.cem_cfg.cem_type in ['POPLINA-INIT', 'POPLINA-REPLAN']:
            self.sol_dim = sol_dim  # add noise ontop of the policy output
        elif self._params.cem_cfg.cem_type == 'POPLINP-SEP':
            # policy network: [dO=17, 64, 64, dU=6], sol_dim = 5568
            # policy network: [dO=17, 64, dU=6], sol_dim = 1472
            # policy network: [dO=17, dU=6], sol_dim = 112 (see ben rechts)
            policy_shape = self._params.cem_cfg.policy_network_shape
            weight_shape = [self.dO] + policy_shape + [self.dU]
            self.sol_dim = 0
            for i_input in range(len(weight_shape) - 1):
                self.sol_dim += \
                    weight_shape[i_input] * weight_shape[i_input + 1]
                self.sol_dim += weight_shape[i_input + 1]

            self._weight_size = self.sol_dim  # passed to the complie_cost()
            self.sol_dim *= self.plan_hor

        elif self._params.cem_cfg.cem_type == 'POPLINP-UNI':
            policy_shape = self._params.cem_cfg.policy_network_shape
            weight_shape = [self.dO] + policy_shape + [self.dU]
            self.sol_dim = 0
            for i_input in range(len(weight_shape) - 1):
                self.sol_dim += \
                    weight_shape[i_input] * weight_shape[i_input + 1]
                self.sol_dim += weight_shape[i_input + 1]

            self._weight_size = self.sol_dim
        else:
            raise NotImplementedError

    def clean_dataset(self):
        self._dataset = []

    def upload_dataset(self):
        # TODO: in the future, we might need several threads.
        # THE datasets should be a data structure in the MPC.py
        data = self._dataset
        self.clean_dataset()
        return data

    def train(self, obs_trajs, acs_trajs, rews_trajs, imaginary_replay_buffer=None):
        # construct the "on policy" data
        if imaginary_replay_buffer is None:
            imaginary_replay_buffer = self._dataset

        # construct the dataset with fake data
        imaginary_dataset = {
            key: [] for key in ['start_state', 'action', 'return', 'weight']
            if key in imaginary_replay_buffer[0]
        }
        for timesteps in range(len(imaginary_replay_buffer)):
            # range the ordering
            '''
            indices = np.argsort(imaginary_replay_buffer[timesteps]['return'])
            # TODO: what about the itermediate steps?
            indices = indices[-self._params.cem_cfg.training_top_k:]
            '''
            for key in imaginary_dataset:
                imaginary_dataset[key].append(
                    imaginary_replay_buffer[timesteps][key]
                )
        for key in imaginary_dataset:
            assert len(imaginary_dataset[key]) > 0
            imaginary_dataset[key] = np.concatenate(imaginary_dataset[key])

        # the dataset with real data
        real_dataset = {
            'start_state': np.concatenate([i_traj[:-1] for i_traj in obs_trajs],
                                          axis=0),
            'action': np.concatenate(acs_trajs, axis=0),
        }
        if 'sol_weight' in imaginary_replay_buffer[0]:
            real_dataset['weight'] = np.array(
                [imaginary_replay_buffer[i]['sol_weight']
                 for i in range(len(imaginary_replay_buffer))]
            )
        real_dataset['state'] = real_dataset['start_state']

        # get the new running mean
        whitening_util.update_whitening_stats(self._whitening_stats,
                                              real_dataset, 'state')
        real_dataset.update({'whitening_stats': self._whitening_stats})

        # train the behavior cloning policy network
        self._policy_network.train(
            real_dataset, training_info={'imaginary_dataset': imaginary_dataset}
        )

        # clear dataset
        self.clean_dataset()

    def forward_policy_propose(self, predict_next_obs, sy_cur_obs):

        with self.tf_sess.graph.as_default():
            self._proposed_act_seqs_ph = None
            self._proposed_act_seqs_res = None

            # first_base_acs is used to recover the first choices
            self._first_base_acs = \
                self._policy_network.forward_network(sy_cur_obs[None])

        if self._params.cem_cfg.cem_type == 'POPLINA-INIT':
            with self.tf_sess.graph.as_default():
                # set up the initial values
                proposed_act_seqs = []
                obs = []
                cur_obs = tf.tile(sy_cur_obs[None],
                                  [self.popsize * self.npart, 1])

                for i_t in range(self.plan_hor):

                    proposed_act = self._policy_network.forward_network(cur_obs)
                    cur_obs = predict_next_obs(cur_obs, proposed_act)
                    obs.append(cur_obs)
                    proposed_act_seqs.append(proposed_act)

                self._proposed_act_seqs_res = tf.stack(proposed_act_seqs)
                self._debug_obs = tf.stack(obs)

        else:
            pass

    def setup(self, cost_function, tf_compatible):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        if not tf_compatible or self.tf_sess is None:
            raise RuntimeError("Cannot pass in a tf.Tensor-valued cost function without passing in a TensorFlow "
                               "session into the constructor")

        self.tf_compatible = tf_compatible

        def continue_optimization(t, mean, var, best_val, best_sol,
                                  elites, returns):
            return tf.logical_and(tf.less(t, self.max_iters),
                                  tf.reduce_max(var) > self.epsilon)

        def iteration(t, mean, var, best_val, best_sol, elites, returns):
            # TODO: no sigmoid at the output?
            samples = tf.truncated_normal([self.popsize, self.sol_dim],
                                          mean, tf.sqrt(var))  # the noise

            costs = cost_function(
                samples, cem_type=self._params.cem_cfg.cem_type,
                tf_data_dict={'policy_network': self._policy_network,
                              'proposed_act_seqs': self._proposed_act_seqs_res}
            )
            values, indices = tf.nn.top_k(-costs, k=self.num_elites,
                                          sorted=True)

            # TODO: how do deal with different particles?
            best_val, best_sol = tf.cond(
                tf.less(-values[0], best_val),
                lambda: (-values[0], samples[indices[0]]),
                lambda: (best_val, best_sol)
            )

            elites = tf.gather(samples, indices)
            returns = -tf.gather(costs, indices)
            new_mean = tf.reduce_mean(elites, axis=0)
            new_var = tf.reduce_mean(tf.square(elites - new_mean), axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            # return t + 1, mean, var, best_val, best_sol, trajs, acs, returns
            return t + 1, mean, var, best_val, best_sol, elites, returns

        with self.tf_sess.graph.as_default():
            self.init_returns = tf.Variable(np.zeros([self.num_elites]),
                                            dtype=tf.float32)

            self.init_elites = tf.tile(self.init_mean[None, :],
                                       [self.num_elites, 1])

            self.num_opt_iters, self.mean, self.var, self.best_val, \
                self.best_sol, self.elites, self.returns = \
                tf.while_loop(cond=continue_optimization, body=iteration,
                              loop_vars=[0, self.init_mean, self.init_var,
                                         float("inf"), self.init_mean,
                                         self.init_elites, self.init_returns])
            self.tf_sess.run(tf.variables_initializer(tf.global_variables()))

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var, per, dU, obs=None):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        assert self.tf_compatible
        sol, solvar, num_opt_iters, elites, returns, \
            start_state, first_base_acs = self.tf_sess.run(
                [self.mean, self.var, self.num_opt_iters,
                 self.elites, self.returns, self.sy_cur_obs, self._first_base_acs],
                feed_dict={self.init_mean: init_mean, self.init_var: init_var}
            )
        '''
        propose_act_seq, propose_obs = self.tf_sess.run(
            [self._proposed_act_seqs_res, self._debug_obs],
            feed_dict={self.init_mean: init_mean, self.init_var: init_var}
        )
        '''

        assert start_state.shape[0] == self.dO
        assert first_base_acs.shape[0] == 1
        assert elites.shape[0] == self.num_elites
        assert returns.shape[0] == self.num_elites

        imaginary_data = {
            # TODO: use data that are not good? check lb size (all -1?)
            'start_state': np.tile(start_state[None, :], [self.num_elites, 1]),
            'action': np.maximum(
                np.minimum(first_base_acs + elites[:, :self.dU],
                           self.ub[0]), self.lb[0]
            ),
            'return': returns.reshape([-1, 1])
        }
        self._dataset.append(imaginary_data)

        prev_sol = self.update_prev_sol(per, dU, sol)
        sol_action = first_base_acs + sol[:self.dU]  # the real control signal

        return sol_action, prev_sol

    def train_policy_network(self):
        return True

    def get_policy_network(self):
        return self._policy_network

    def obtain_test_solution(self, init_mean, init_var, per, dU, obs=None, average=False):
        """Optimizes the cost function using the provided initial candidate distribution
        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        if average:
            raise NotImplementedError
        else:
            sol = np.zeros(self.sol_dim)

            # the actualy solution
            # first_acs = self.tf_sess.run(self._first_base_sol)
            first_acs = self.tf_sess.run(self._first_base_acs)

            sol_action = first_acs.reshape([-1])  # the actual action to be used
            prev_sol = self.update_prev_sol(per, dU, sol)

        return sol_action, prev_sol
