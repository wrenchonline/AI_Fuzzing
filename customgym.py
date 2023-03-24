import gym
from gym import spaces
import numpy as np
from gym.envs.registration import register


class CustomEnv(gym.Env):
    def __init__(self):
        # Define the state space and action space
        self.observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(2)
        # Define any other variables needed
        self.current_state = 0
        self.steps_taken = 0

    def step(self, action):
        # Execute the action
        reward = 0
        if action == 1:
            self.current_state += 1
        # Calculate the reward
        if self.current_state == 9:
            reward = 1
        # Check if episode is done
        done = (self.current_state == 9 or self.steps_taken == 50)
        self.steps_taken += 1
        # Return the next state, reward, and done flag
        return self.current_state, reward, done, {}

    def reset(self):
        # Reset the environment
        self.current_state = 0
        self.steps_taken = 0
        return self.current_state


register(
    id='CustomEnv-v0',
    entry_point='custom_env:CustomEnv',
)
