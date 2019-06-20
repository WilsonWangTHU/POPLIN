from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from .policy_network import gmm_policy


from .pocem import POPLIN-AOptimizer


class PGCEMOptimizer(POPLIN-AOptimizer):
    """A Tensorflow-compatible CEM optimizer.
        Model the policy based CEM with Gaussian / Mixture of Gaussian prior.

        1. CEM strategy and Noise strategy:
            ## PGW
            @PGWCEM-TRAJ: Similar to PGMMWCEM-ALL, but instead of using GMM, use
                a single Gaussian distribution (can be regarded as Outer loop
                cem)
                initial w1 = w2 = ... = wN ~ p(w);

            @PGWCEM-ALL:
                initial p(w1) = p(w2) = ... = p(wN) ~ p(w);

            ## PGMM
            @PGMMWCEM-TRAJ: Observe the current init_ob, get the weight from
                conditional gaussian.
                initial w1 = w2 = ... = wN ~ p(w | o1);

            @PGMMWCEM-ALLSP: the GMM version, but with the same distribution
                initial p(w1) = p(w2) = ... = p(wN) ~ p(w | o1);

            @PGMMWCEM-ALLM: Have multiple GMM, with different GMM at different
                timesteps during planning
                initial w1 ~ p(w1| o1); w2 ~ p(w2 | o1); ...;  wN ~ p(wN | o1);

            # TODO: does tensorflow support this?
            @PGMMWCEM-ALLR: Replannning by looking at the predicted observation
                initial w1 ~ p(w| o1); w2 ~ p(w | o2); ...;  wN ~ p(w | oN);

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
        self.max_iters, self.popsize, self.num_elites = \
            max_iters, popsize, num_elites
        self._params = params
        self._cem_type = params.cem_cfg.cem_type
        self.set_sol_dim(sol_dim)
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha
        self.tf_sess = tf_session
        self._dataset = []

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

        if self._params.cem_cfg.training_scheme in ['G-R', 'G-I',
                                                    'GMM-R', 'GMM-I']:
            # behavior cloning network
            self._params.cem_cfg.plan_hor = self.plan_hor
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("gaussian_policy_network"):
                    self._policy_network = gmm_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_gaussian_network',
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

        if self._params.cem_cfg.cem_type in ['PGWCEM-ALL', 'PGMMWCEM-ALLSP',
                                             'PGMMWCEM-ALLM', 'PGMMWCEM-ALLR']:
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

        elif self._params.cem_cfg.cem_type in ['PGWCEM-TRAJ', 'PGMMCEM-TRAJ']:
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

    def train(self, obs_trajs, acs_trajs, rews_trajs, imaginary_replay_buffer=None):
        raise NotImplementedError
        """
        # construct the "on policy" data
        if imaginary_replay_buffer is None:
            imaginary_replay_buffer = self._dataset

        # construct the dataset with fake data
        imaginary_dataset = {
            key: [] for key in ['start_state', 'action', 'return', 'weight']
            if key in imaginary_replay_buffer[0]
        }
        for timesteps in range(len(imaginary_replay_buffer)):
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

        # train the behavior cloning policy network
        self._policy_network.train(
            real_dataset, training_info={'imaginary_dataset': imaginary_dataset}
        )

        # clear dataset
        self.clean_dataset()
        """

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
            return tf.less(t, self.max_iters)

        def iteration(t, mean, var, best_val, best_sol, elites, returns):
            samples = tf.truncated_normal([self.popsize, self.sol_dim],
                                          mean, tf.sqrt(var))  # the noise

            if self._cem_type == 'POPLINP-UNI':
                # duplicate the weight noise for every steps (plan_hor)
                weight_noise = tf.tile(samples[:, None, :], [1, self.plan_hor, 1])
            elif self._cem_type == 'POPLINP-SEP':
                # reshape
                weight_noise = tf.reshape(samples, [self.popsize, self.plan_hor, -1])
            else:
                raise NotImplementedError

            costs = cost_function(
                weight_noise, cem_type=self._params.cem_cfg.cem_type,
                tf_data_dict={'policy_network': self._policy_network,
                              'weight_size': self._weight_size}
            )
            values, indices = tf.nn.top_k(-costs, k=self.num_elites,
                                          sorted=True)

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
            '''
            self.num_opt_iters, self.mean, self.var, self.best_val, \
                self.best_sol, self.elites, self.returns = \
                iteration(tf.constant(0), self.init_mean, self.init_var,
                        float("inf"), self.init_mean,
                        self.init_elites, self.init_returns)
            '''
            self.tf_sess.run(tf.variables_initializer(tf.global_variables()))

    def obtain_solution(self, init_mean, init_var, per, dU, obs=None):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        assert self.tf_compatible
        if self._cem_type == 'PGWCEM-TRAJ':
            """ @PGWCEM-TRAJ: Similar to PGMMWCEM-ALL, but instead of using GMM, use
                a single Gaussian distribution (can be regarded as Outer loop
                cem)
                initial w1 = w2 = ... = wN ~ p(w);
            """

        elif self._cem_type == 'PGWCEM-ALL':
            """ @PGWCEM-ALL: initial p(w1) = p(w2) = ... = p(wN) ~ p(w);
            """
            pass

        elif self._cem_type == 'PGMMWCEM-TRAJ':
            """ @PGMMWCEM-TRAJ: Observe the current init_ob, get the weight from
                conditional gaussian.
                initial w1 = w2 = ... = wN ~ p(w | o1);
            """
            pass

        elif self._cem_type == 'PGMMWCEM-ALLSP':
            """ @PGMMWCEM-ALLSP: the GMM version, but with the same distribution
                initial p(w1) = p(w2) = ... = p(wN) ~ p(w | o1);
            """
            pass

        elif self._cem_type == 'PGMMWCEM-ALLM':
            """ @PGMMWCEM-ALLM: Have multiple GMM, with different GMM at
                different timesteps during planning
                initial w1 ~ p(w1| o1); w2 ~ p(w2 | o1); ...;  wN ~ p(wN | o1);
            """
            pass

        elif self._cem_type == 'PGMMWCEM-ALLR':
            """ @PGMMWCEM-ALLR: Replannning by looking at the predicted observation
                initial w1 ~ p(w| o1); w2 ~ p(w | o2); ...;  wN ~ p(w | oN);
                # TODO: does tensorflow support this?
            """
            raise NotImplementedError

        init_var = np.ones(self.sol_dim) * np.mean(init_var)
        sol, solvar, num_opt_iters, elites, returns, start_state = \
            self.tf_sess.run(
                [self.mean, self.var, self.num_opt_iters,
                 self.elites, self.returns, self.sy_cur_obs],
                feed_dict={
                    self.init_mean:
                        init_mean * self._params.cem_cfg.pwcem_init_mean,
                    self.init_var: init_var
                }
            )

        # the actualy solution
        first_acs = self.tf_sess.run(
            self._first_base_sol,
            feed_dict={self._sol_weight_input: np.reshape(
                sol[:self._weight_size], [1, -1]
            )}
        )

        imaginary_data = {
            'start_state': np.tile(start_state[None, :], [self.num_elites, 1]),
            'weight': elites[:, :self._weight_size],
            'return': returns.reshape([-1, 1]),
            'sol_weight': sol[:self._weight_size]
        }

        assert start_state.shape[0] == self.dO
        assert first_acs.shape[0] == 1
        assert elites.shape[0] == self.num_elites
        assert returns.shape[0] == self.num_elites

        self._dataset.append(imaginary_data)

        sol_action = first_acs.reshape([-1])  # the actual action to be used
        prev_sol = self.update_prev_sol(per, dU, sol)

        return sol_action, prev_sol

    def update_prev_sol(self, per, dU, soln):
        """ @brief: in pwcem, the soln is not the actual search-space candiates
            sol --> self.sol_dim
        """
        if self._cem_type == 'POPLINP-UNI':
            prev_sol = soln

        elif self._cem_type == 'POPLINP-SEP':
            prev_sol = np.concatenate([np.copy(soln)[per * self._weight_size:],
                                       np.zeros(per * self._weight_size)])

        else:
            raise NotImplementedError

        return prev_sol

    def reset_prev_sol(self, prev_sol):
        prev_sol = np.mean(prev_sol) * np.ones([self.sol_dim])
        return prev_sol

    def forward_policy_propose(self, predict_next_obs, sy_cur_obs):
        """ @brief: for pwcem, we only need to get the first actions
        """
        with self.tf_sess.graph.as_default():
            self._sol_weight_input = tf.placeholder(
                dtype=tf.float32, shape=[1, self._weight_size]
            )
            self._first_base_sol = self._policy_network.forward_network(
                sy_cur_obs[None], self._sol_weight_input
            )

    def train_policy_network(self):
        return True
