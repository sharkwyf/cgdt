# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import gym
import numpy as np
import pickle

from gym import spaces
from gym.utils import seeding


class BernoulliBanditEnv(gym.Env):

  def __init__(self,
               num_arms=2,
               reward_power=3.0,
               reward_scale=0.9,
               generation_seed=0,
               bernoulli_prob=None,
               loop=False):
    self._num_arms = num_arms
    self._reward_power = reward_power
    self._reward_scale = reward_scale
    self._bernoulli_prob = bernoulli_prob
    self._loop = loop
    self._generate_bandit(generation_seed)

    self.observation_space = spaces.Discrete(1)
    self.action_space = spaces.Discrete(self._num_arms)

    self.seed()
    self.reset()

  def _generate_bandit(self, seed):
    gen_random, _ = seeding.np_random(seed)

    if self._bernoulli_prob and self._num_arms == 2:
      self._rewards = np.asarray(
          [self._bernoulli_prob, 1 - self._bernoulli_prob])
    else:
      self._rewards = gen_random.random_sample([self._num_arms])
      self._rewards = self._reward_scale * self._rewards**self._reward_power

  @property
  def rewards(self):
    return self._rewards

  @property
  def num_arms(self):
    return self._num_arms

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    return self._get_obs()

  def _get_obs(self):
    return np.zeros((1))

  def step(self, action):
    reward = self._rewards[action]
    sampled_reward = float(self.np_random.random_sample() <= reward)
    done = not self._loop
    return self._get_obs(), sampled_reward, done, {}


class BernoulliBanditWrapper(gym.Wrapper):

  def __init__(self, env):
    super().__init__(env)

  def step(self, action):
    action = np.random.choice(np.arange(2), p=action)
    # action = np.argmax(action, axis=-1)
    obs, reward, done, info = super().step(action)
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    return obs


def get_bandit_policy(bandit_env,
                      epsilon_explore=0.0,
                      py=True,
                      return_distribution=True,
                      bernoulli_prob=None):
  """Creates an optimal policy for solving the bandit environment.

  Args:
    bandit_env: A bandit environment.
    epsilon_explore: Probability of sampling random action as opposed to optimal
      action.
    py: Whether to return Python policy (NumPy) or TF (Tensorflow).
    return_distribution: In the case of a TF policy, whether to return the
      full action distribution.

  Returns:
    A policy_fn that takes in an observation and returns a sampled action along
      with a dictionary containing policy information (e.g., log probability).
    A spec that determines the type of objects returned by policy_info.

  Raises:
    ValueError: If epsilon_explore is not a valid probability.
  """
  if epsilon_explore < 0 or epsilon_explore > 1:
    raise ValueError('Invalid exploration value %f' % epsilon_explore)

  if bernoulli_prob and bandit_env.num_arms == 2:
    policy_distribution = np.asarray([[1 - bernoulli_prob, bernoulli_prob]])
  else:
    optimal_action = np.argmax(bandit_env.rewards)
    policy_distribution = np.ones([1, bandit_env.num_arms
                                  ]) / bandit_env.num_arms
    policy_distribution[0] *= epsilon_explore
    policy_distribution[0, optimal_action] += 1 - epsilon_explore

  def obs_to_index_fn(observation):
    if py:
      return np.array(observation, dtype=np.int32)
    else:
      return tf.cast(observation, tf.int32)

  if py:
    return common_utils.create_py_policy_from_table(
        policy_distribution, obs_to_index_fn)
  else:
    return common_utils.create_tf_policy_from_table(
        policy_distribution, obs_to_index_fn,
        return_distribution=return_distribution)


def get_bandit_dataset(bandit_env,
                       p=None,
                       N_steps=1e3):
    name = "bernoulli-bandit"
    paths = []
    collected_steps = 0
    while collected_steps < N_steps:
        obs_list = [bandit_env.reset()]
        actions = []
        rewards = []
        dones = []
        infos = []
        
        for i in range(1):
            if p is None:
                action = bandit_env.action_space.sample()
            else:
                action = np.random.choice(np.arange(2), p=[p, 1-p])
                action = np.eye(2)[action]
            next_obs, reward, done, info = bandit_env.step(action)

            obs_list.append(next_obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        episode_data = {
            "observations": np.array(obs_list[:-1]),
            "next_observations": np.array(obs_list[1:]),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "terminals": np.array(dones),
        }
        paths.append(episode_data)
        collected_steps += len(rewards)

    returns = np.array([np.sum(p["rewards"]) for p in paths])
    num_samples = np.sum([p["rewards"].shape[0] for p in paths])
    print(f"{name}: Number of samples collected: {num_samples}")
    print(
        f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
    )

    with open(f"{name}-p{p}.pkl", "wb") as f:
        pickle.dump(paths, f)

    return None


if __name__ == "__main__":
    if False:
        env = BernoulliBanditEnv(
            num_arms=2,
            reward_power=3.0,
            reward_scale=0.9,
            generation_seed=0,
            bernoulli_prob=0.9,
            loop=False,
        )
        env = BernoulliBanditWrapper(env)
        res = {0: [], 1: []}

        env.reset()
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            res[action].append(reward)
            print(action, obs, reward, done, info)
            pass
        
        for k, v in res.items():
            print(f"{k}: {sum(res[k])} / {len(res[k])}")
        
        print("Done")
    else:
        for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
            env = BernoulliBanditEnv(
                num_arms=2,
                reward_power=3.0,
                reward_scale=0.9,
                generation_seed=0,
                bernoulli_prob=1-p,
                loop=False,
            )
            env = BernoulliBanditWrapper(env)
            get_bandit_dataset(env, p, N_steps=1e4)