from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np


class Optimizer:

    def __init__(self, *args, **kwargs):
        self.sy_cur_obs = None
        self._proposed_act_seqs_ph = None
        pass

    def setup(self, cost_function, tf_compatible):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")

    def get_policy_network(self):
        return None

    def train_policy_network(self):
        return False

    def set_sy_cur_obs(self, sy_cur_obs):
        # NOTE: it is a hack! be careful
        self.sy_cur_obs = sy_cur_obs

    def forward_policy_propose(self, predict_next_obs, sy_cur_obs):
        pass

    def reset_prev_sol(self, prev_sol):
        return prev_sol

    def update_prev_sol(self, per, dU, soln):
        prev_sol = np.concatenate([np.copy(soln)[per * dU:],
                                   np.zeros(per * dU)])
        return prev_sol
