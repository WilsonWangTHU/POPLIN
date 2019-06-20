from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.stats as stats
from dmbrl.misc import logger

from .optimizer import Optimizer


class GBPCEMOptimizer(Optimizer):
    """ @brief: The basic idea is too add a gradient based optimization after /
        before the top k is chosen

        @gbp_type
        @1 / 2: do the gradient based-planning with in the loop
            @1: do the planning for all the candidates
            @2: do the planning only for the top k candidates

        @3: do the gradient based planning only after we get the mean (in numpy)
    """
    """A Tensorflow-compatible gradient based policy optimizer.
        To get started, it is the first applied to the random shooting algorithm.
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
        self._params = params
        self._print_count = 0
        self._gbp_type = self._params.gbp_cfg.gbp_type  # see @gbp_type on top
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha
        self.tf_sess = tf_session
        self._tf_dict = {}

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        if self.tf_sess is not None:
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("CEMSolver"):
                    self.init_mean = tf.placeholder(dtype=tf.float32, shape=[sol_dim])
                    self.init_var = tf.placeholder(dtype=tf.float32, shape=[sol_dim])

        self.num_opt_iters, self.mean, self.var = None, None, None
        self.tf_compatible, self.cost_function = None, None

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

        self.tf_compatible = tf_compatible

        if self._gbp_type == 3:
            # @3: do the gradient based planning only after we get the mean (in numpy)
            def continue_optimization(t, mean, var, best_val, best_sol):
                return tf.logical_and(tf.less(t, self.max_iters), tf.reduce_max(var) > self.epsilon)

            def iteration(t, mean, var, best_val, best_sol):
                lb_dist, ub_dist = mean - self.lb, self.ub - mean
                constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
                samples = tf.truncated_normal([self.popsize, self.sol_dim], mean, tf.sqrt(constrained_var))

                costs = cost_function(samples)
                values, indices = tf.nn.top_k(-costs, k=self.num_elites, sorted=True)

                best_val, best_sol = tf.cond(
                    tf.less(-values[0], best_val),
                    lambda: (-values[0], samples[indices[0]]),
                    lambda: (best_val, best_sol)
                )

                elites = tf.gather(samples, indices)
                new_mean = tf.reduce_mean(elites, axis=0)
                new_var = tf.reduce_mean(tf.square(elites - new_mean), axis=0)

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                return t + 1, mean, var, best_val, best_sol

            with self.tf_sess.graph.as_default():
                self.num_opt_iters, self.mean, self.var, self.best_val, self.best_sol = tf.while_loop(
                    cond=continue_optimization, body=iteration,
                    loop_vars=[0, self.init_mean, self.init_var, float("inf"), self.init_mean]
                )

                self.set_planning_network(cost_function, 1, 'mean')
        elif self._gbp_type == 2:
            '''
                @1 / 2: do the gradient based-planning with in the loop
                    @1: do the planning for all the candidates
                    @2: do the planning only for the top k candidates
            '''
            self.set_planning_network(cost_function, self.popsize, 'popsize')
            self.set_planning_network(cost_function, self.num_elites, 'top_k')
        elif self._gbp_type == 1:
            '''
                @1 / 2: do the gradient based-planning with in the loop
                    @1: do the planning for all the candidates
                    @2: do the planning only for the top k candidates
            '''
            self.set_planning_network(cost_function, self.popsize, 'popsize')

    def reset(self):
        pass

    def set_planning_network(self, cost_function, candidate_size, name):
        tf_dict = {}
        tf_dict['candidate_solutions'] = tf.Variable(
            np.random.uniform(self.lb, self.ub, [candidate_size, self.sol_dim]),
            dtype=tf.float32
        )
        self.tf_sess.run(
            tf.variables_initializer([tf_dict['candidate_solutions']])
        )

        tf_dict['costs'] = costs = cost_function(tf_dict['candidate_solutions'])
        tf_dict['choice'] = tf.argmin(costs)
        tf_dict['solution'] = \
            tf_dict['candidate_solutions'][tf.cast(tf_dict['choice'], tf.int32)]

        # the update loss
        tf_dict['adam_optimizer'] = \
            tf.train.AdamOptimizer(learning_rate=self._params.gbp_cfg.lr)
        tf_dict['planning_optimizer'] = tf_dict['adam_optimizer'].minimize(
            costs, var_list=[tf_dict['candidate_solutions']]
        )
        self.tf_sess.run(
            tf.variables_initializer(tf_dict['adam_optimizer'].variables())
        )
        self._tf_dict[name] = tf_dict

    def obtain_solution(self, init_mean, init_var, per, dU, obs=None):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """

        self._print_count = (self._print_count + 1) % 20
        self._print = self._print_count == 0

        if self._gbp_type == 3:
            sol, solvar = self.tf_sess.run(
                [self.mean, self.var],
                feed_dict={self.init_mean: init_mean, self.init_var: init_var}
            )

            self._tf_dict['mean']['candidate_solutions'].load(
                sol.reshape([1, -1]), self.tf_sess
            )

            avg_cost = self.tf_sess.run(self._tf_dict['mean']['costs']).reshape([-1])
            if self._print:
                logger.info('Init   -> cost: %.3f' % (avg_cost))

            # step 2: do gradient based planning
            for gbp_iteration in range(self._params.gbp_cfg.plan_iter):
                self.tf_sess.run(self._tf_dict['mean']['planning_optimizer'])

            avg_cost = self.tf_sess.run(self._tf_dict['mean']['costs']).reshape([-1])
            if self._print:
                logger.info('AFTER %d iter -> cost: %.3f' %
                            (self._params.gbp_cfg.plan_iter, avg_cost))
            sol = self.tf_sess.run(self._tf_dict['mean']['solution']).reshape([-1])

        elif self._gbp_type == 2:
            '''
                @1 / 2: do the gradient based-planning with in the loop
                    @1: do the planning for all the candidates
                    @2: do the planning only for the top k candidates
            '''
            mean, var, t = init_mean, init_var, 0
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean),
                                scale=np.ones_like(mean))

            while (t < self.max_iters) and np.max(var) > self.epsilon:
                lb_dist, ub_dist = mean - self.lb, self.ub - mean
                constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2),
                                             np.square(ub_dist / 2)), var)

                samples = X.rvs(size=[self.popsize, self.sol_dim]) * \
                    np.sqrt(constrained_var) + mean
                self._tf_dict['popsize']['candidate_solutions'].load(
                    samples.reshape([self.popsize, -1]), self.tf_sess
                )
                costs = self.tf_sess.run(self._tf_dict['popsize']['costs'])
                sort_id = np.argsort(costs)
                elites = samples[sort_id][:self.num_elites]

                # step 2: do gradient based planning for the top k candidates
                self._tf_dict['top_k']['candidate_solutions'].load(
                    elites.reshape([self.num_elites, -1]), self.tf_sess
                )

                if self._print:
                    logger.info('Init elites score  -> cost: %f' %
                                np.mean(costs[sort_id][:self.num_elites]))
                for gbp_iteration in range(self._params.gbp_cfg.plan_iter):
                    self.tf_sess.run(
                        self._tf_dict['top_k']['planning_optimizer']
                    )

                if self._print:
                    logger.info('AFTER %d iter -> cost: %f.' % (
                        self._params.gbp_cfg.plan_iter,
                        np.mean(self.tf_sess.run(self._tf_dict['top_k']['costs']))
                    ))

                elites = self.tf_sess.run(
                    self._tf_dict['top_k']['candidate_solutions']
                )

                new_mean = np.mean(elites, axis=0)
                new_var = np.var(elites, axis=0)

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                t += 1
            sol, solvar = mean, var

        elif self._gbp_type == 1:
            '''
                @1 / 2: do the gradient based-planning with in the loop
                    @1: do the planning for all the candidates
                    @2: do the planning only for the top k candidates
            '''

            mean, var, t = init_mean, init_var, 0
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean),
                                scale=np.ones_like(mean))

            while (t < self.max_iters) and np.max(var) > self.epsilon:
                lb_dist, ub_dist = mean - self.lb, self.ub - mean
                constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2),
                                             np.square(ub_dist / 2)), var)

                samples = X.rvs(size=[self.popsize, self.sol_dim]) * \
                    np.sqrt(constrained_var) + mean
                self._tf_dict['popsize']['candidate_solutions'].load(
                    samples.reshape([self.popsize, -1]), self.tf_sess
                )

                costs = self.tf_sess.run(self._tf_dict['popsize']['costs'])
                sort_id = np.argsort(costs)
                old_elites = samples[sort_id][:self.num_elites]
                old_costs = costs[sort_id][:self.num_elites]
                if self._print:
                    logger.info('Init elites score  -> cost: %f' %
                                np.mean(old_costs))

                # step 2: do gradient based planning
                for gbp_iteration in range(self._params.gbp_cfg.plan_iter):
                    self.tf_sess.run(
                        self._tf_dict['popsize']['planning_optimizer']
                    )

                samples, costs = self.tf_sess.run(
                    [self._tf_dict['popsize']['candidate_solutions'],
                     self._tf_dict['popsize']['costs']]
                )

                elites = np.concatenate([samples, old_elites], axis=0)
                costs = np.concatenate([costs, old_costs])
                sort_id = np.argsort(costs)
                elites = elites[sort_id][:self.num_elites]
                costs = costs[sort_id][:self.num_elites]

                if self._print:
                    logger.info('AFTER %d iter -> cost: %f.' %
                                (self._params.gbp_cfg.plan_iter, np.mean(costs)))

                new_mean = np.mean(elites, axis=0)
                new_var = np.var(elites, axis=0)

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                t += 1
            sol, solvar = mean, var

        else:
            assert False

        prev_sol = self.update_prev_sol(per, dU, sol)
        return sol, prev_sol
