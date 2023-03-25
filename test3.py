import gym
from gym import spaces
import numpy as np

a = np.array([1, 4, 3, 4])
# 定义一个gym space对象
a_space = spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32)


print(a_space.sample())  # Output: [1, 4, 3, 4] randomly
