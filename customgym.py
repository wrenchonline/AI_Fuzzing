import gym
from gym import spaces
import numpy as np


class MyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = spaces.Discrete(2)
        self.seed()
        self.reset()
        self.last_action = None
        self.last_reward = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.array([0.5, 0.5])
        self.last_action = None
        self.last_reward = None
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        if action == 0:
            self.state[0] -= 0.1
        else:
            self.state[0] += 0.1
        self.state[1] = np.sin(self.state[0])
        reward = 1.0 if abs(self.state[1]) < 0.1 else -1.0
        done = abs(self.state[1]) < 0.1
        info = {}

        if action == self.last_action:
            reward = self.last_reward

        self.last_action = action
        self.last_reward = reward

        return self.state, reward, done, info


gym.register(
    id='MyEnv-v0',
    entry_point='myenv:MyEnv',
)
