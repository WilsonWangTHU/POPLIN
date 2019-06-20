from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
"""
    Module name,
    MODEL_IN, MODEL_OUT,
    import env, env_name
"""


class GymAcrobotConfigModule:
    ENV_NAME = "MBRLGYM_acrobot-v0"
    TASK_HORIZON = 1000
    NTRAIN_ITERS = 300
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 30
    INIT_VAR = 0.25
    MODEL_IN, MODEL_OUT = 7, 6  # obs -> 6, action -> 1
    GP_NINDUCING_POINTS = 300

    def __init__(self):
        # self.ENV = gym.make(self.ENV_NAME)
        from mbbl.env.gym_env import acrobot
        self.ENV = acrobot.env(env_name='gym_acrobot', rand_seed=1234,
                               misc_info={'reset_type': 'gym'})
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "GBPRandom": {
                "popsize": 2500
            },
            "GBPCEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "POPLIN-P": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "POPLIN-A": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        """ @brief: no cheating of the observation function
        """
        if isinstance(obs, np.ndarray):
            return obs
        else:
            return obs

    @staticmethod
    def obs_postproc(obs, pred):
        if isinstance(obs, np.ndarray):
            return obs + pred
        else:
            return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        """ @brief:

            def reward(data_dict):
                def height(obs):
                    h1 = obs[0]  # Height of first arm
                    h2 = obs[0] * obs[2] - obs[1] * obs[3]  # Height of second arm
                    return -(h1 + h2)  # total height

                start_height = height(data_dict['start_state'])

                reward = {
                    'gym_acrobot': start_height,
                    'gym_acrobot_sparse': (start_height > 1) - 1
                }[self._env_name]  # gets gt reward based on sparse/dense
                return reward
            self.reward = reward
        """
        return obs[:, 0] + obs[:, 0] * obs[:, 2] - obs[:, 1] * obs[:, 3]

    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return np.sum(np.square(acs), axis=1) * 0.0
        else:
            return tf.reduce_sum(tf.square(acs), axis=1) * 0.0

    def nn_constructor(self, model_init_cfg, misc=None):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None),
            misc=misc
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.000025))
            model.add(FC(200, activation="swish", weight_decay=0.00005))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0001))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model


CONFIG_MODULE = GymAcrobotConfigModule
