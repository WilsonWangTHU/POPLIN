from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


class Controller:
    def __init__(self, *args, **kwargs):
        """Creates class instance.
        """
        self._policy_network = None
        pass

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains this controller using lists of trajectories.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        """Resets this controller.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def act(self, obs, t, get_pred_cost=False):
        """Performs an action.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def dump_logs(self, primary_logdir, iter_logdir):
        """Dumps logs into primary log directory and per-train iteration log directory.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def get_policy_network(self):
        return self._policy_network
    
    def train_policy_network(self):
        return False
