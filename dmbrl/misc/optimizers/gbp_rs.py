from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .optimizer import Optimizer
from dmbrl.misc import logger


class GBPRandomOptimizer(Optimizer):
    """ @brief: use gradient based planning to update the policy network
    """

    def __init__(self, sol_dim, popsize, tf_session,
                 upper_bound=None, lower_bound=None, params=None):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            tf_session (tf.Session): (optional) Session to be used for this optimizer. Defaults to None,
                in which case any functions passed in cannot be tf.Tensor-valued.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
        """
        super().__init__()
        self._params = params
        self._print_count = 0

        self.sol_dim = sol_dim
        self.popsize = popsize
        self.ub, self.lb = upper_bound, lower_bound
        self.tf_sess = tf_session
        self.solution = None
        self.tf_compatible, self.cost_function = None, None

        self._debug = {}
        self._debug['old_sol'] = 0.0
        self._debug_start = False

    def setup(self, cost_function, tf_compatible):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        if tf_compatible and self.tf_sess is None:
            raise RuntimeError("Cannot pass in a tf.Tensor-valued cost function without passing in a TensorFlow "
                               "session into the constructor")

        if not tf_compatible:
            self.tf_compatible = False
            self.cost_function = cost_function
        else:
            with self.tf_sess.graph.as_default():
                self.tf_compatible = True
                self._candidate_solutions = tf.Variable(
                    np.random.uniform(self.lb, self.ub, [self.popsize, self.sol_dim]),
                    dtype=tf.float32
                )
                self.tf_sess.run(
                    tf.variables_initializer([self._candidate_solutions])
                )

                self._costs = costs = cost_function(self._candidate_solutions)
                self._choice = tf.argmin(costs)
                self.solution = \
                    self._candidate_solutions[tf.cast(self._choice, tf.int32)]

                # the update loss
                self._adam_optimizer = \
                    tf.train.AdamOptimizer(learning_rate=self._params.gbp_cfg.lr)
                self._planning_optimizer = self._adam_optimizer.minimize(
                    costs, var_list=[self._candidate_solutions]
                )
                self.tf_sess.run(
                    tf.variables_initializer(self._adam_optimizer.variables())
                )
                self._average_cost = tf.reduce_mean(costs)
                self._min_cost = tf.reduce_min(costs)
                self._values, self._indices = tf.nn.top_k(-costs, k=10, sorted=True)

                # debug information
                self._debug_actions = self.solution

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var, per, dU, obs=None):
        """Optimizes the cost function provided in setup().
            do gradient based planning

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        assert self.tf_compatible
        self._print_count = (self._print_count + 1) % 20
        self._print = self._print_count == 0

        # step 1: initialize the action candidates TODO: use init_mean
        self._old_solutions = np.concatenate(
            [self.tf_sess.run(self._candidate_solutions)[:, 6:],
             np.random.uniform(self.lb[0], self.ub[0], [self.popsize, 6])],
            axis=1
        )
        self._candidate_solutions.load(self._old_solutions, self.tf_sess)

        avg_cost, min_cost = self.tf_sess.run(
            [self._average_cost, self._min_cost]
        )
        if self._print:
            logger.info('Init   -> Avg_cost: %.3f, Min_cost: %.3f' %
                        (avg_cost, min_cost))

        # step 2: do gradient based planning
        for gbp_iteration in range(self._params.gbp_cfg.plan_iter):
            _, avg_cost, min_cost = self.tf_sess.run(
                [self._planning_optimizer, self._average_cost, self._min_cost]
            )
        avg_cost, min_cost = self.tf_sess.run(
            [self._average_cost, self._min_cost]
        )
        if self._print:
            logger.info('Iter %d > Avg_cost: %.3f, Min_cost: %.3f' %
                        (self._params.gbp_cfg.plan_iter, avg_cost, min_cost))

        sol = self.tf_sess.run(self.solution)
        prev_sol = self.update_prev_sol(per, dU, sol)

        return sol, prev_sol
