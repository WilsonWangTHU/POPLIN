from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from gym.monitoring import VideoRecorder
from dotmap import DotMap
from dmbrl.misc import logger

import time


class Agent:
    """An general class for RL agents.
    """

    def __init__(self, params):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        self.env = params.env

        # load the imitation data if needed
        if hasattr(self.env, '_expert_data_loaded') and \
                (not self.env._expert_data_loaded):
            self.env.load_expert_data(
                params.params.misc.ctrl_cfg.il_cfg.expert_amc_dir
            )

        self.noise_stddev = params.noise_stddev if params.get("noisy_actions", False) else None

        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")
        if (not isinstance(self.noise_stddev, float)) and params.get("noisy_actions", False):
            raise ValueError("Must provide standard deviation for noise for noisy actions.")

        if self.noise_stddev is not None:
            self.dU = self.env.action_space.shape[0]
        self._debug = 1

    def sample(self, horizon, policy, record_fname=None, test_policy=False, average=False):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        if test_policy:
            logger.info('Testing the policy')
        video_record = record_fname is not None
        recorder = None if not video_record else VideoRecorder(self.env, record_fname)

        times, rewards = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False
        self._debug += 1

        policy.reset()
        # for t in range(20):
        for t in range(horizon):
            if hasattr(self.env, 'render_imitation'):
                self.env.render_imitation()
            if t % 50 == 10 and t > 1:
                logger.info('Current timesteps: %d / %d, average time: %.5f'
                            % (t, horizon, np.mean(times)))
            if video_record:
                recorder.capture_frame()
            start = time.time()
            if test_policy:
                A.append(policy.act(O[t], t, test_policy=test_policy, average=average))
            else:
                A.append(policy.act(O[t], t))
            times.append(time.time() - start)

            if self.noise_stddev is None:
                obs, reward, done, info = self.env.step(A[t])
            else:
                action = A[t] + np.random.normal(loc=0, scale=self.noise_stddev,
                                                 size=[self.dU])
                action = np.minimum(np.maximum(action,
                                               self.env.action_space.low),
                                    self.env.action_space.high)
                obs, reward, done, info = self.env.step(action)
            O.append(obs)
            reward_sum += reward
            rewards.append(reward)
            if done:
                break

        if video_record:
            recorder.capture_frame()
            recorder.close()

        logger.info("Average action selection time: %.4f" % np.mean(times))
        logger.info("Rollout length: %d" % len(A))

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }
