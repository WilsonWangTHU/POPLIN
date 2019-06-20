from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
# from dmbrl.misc.DotmapUtils import get_required_argument

# import scipy.stats as stats

from .POPLIN_A import POPLINAOptimizer


class POPLINPOptimizer(POPLINAOptimizer):
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

        2. training_schemes
            @BC-R: behavior cloning training only with the real data
            @BC-AI: behavior cloning training, train towards the action (added
                by the noise) during planning (imaginary_dataset).
            @BC-WI: behavior cloning training, train towards the weight (added
                by the noise) during the planning (imaginary_dataset).

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
        super(POPLINPOptimizer, self).__init__(
            sol_dim, max_iters, popsize, num_elites, tf_session,
            upper_bound, lower_bound, epsilon, alpha, params
        )

        # TODO: the weight input placeholder

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
            self.tf_sess.run(tf.variables_initializer(tf.global_variables()))

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
            first_acs = self.tf_sess.run(
                self._first_base_sol,
                feed_dict={self._sol_weight_input: np.reshape(
                    sol[:self._weight_size], [1, -1]
                )}
            )

            sol_action = first_acs.reshape([-1])  # the actual action to be used
            prev_sol = self.update_prev_sol(per, dU, sol)

        return sol_action, prev_sol

    def obtain_solution(self, init_mean, init_var, per, dU, obs=None):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        assert self.tf_compatible
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
