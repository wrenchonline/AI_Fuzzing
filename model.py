import random
import numpy as np
#import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from customgym import MyEnv


# 神经网络模型
class DQN(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 定义经验回放缓冲区
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))
        # newstate = np.array(state)

        # newnext_state = np.array(next_state)
        # print(newnext_state.shape)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# 定义动作选择策略
def epsilon_greedy_policy(state, epsilon, action_dim, device, Q):
    if np.random.rand() < epsilon:
        return np.random.choice(action_dim)
    else:
        state = torch.tensor(
            state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = Q(state).argmax(dim=-1)
            # 我设置（1, 1, 1），所以在 [1, 3, 20] 索引就是[1-1 = 0, 3-1 = 2]
            return q_values[0, 2].item()


def train(env_name, num_episodes, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, target_update, lr):
    env = MyEnv()
    obs_dim = env.observation_space.shape[1]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Q = DQN(obs_dim, action_dim).to(device)
    Q_target = DQN(obs_dim, action_dim).to(device)
    Q_target.load_state_dict(Q.state_dict())
    optimizer = optim.Adam(Q.parameters(), lr=lr)
    buffer = ReplayBuffer(10000)

    episode_rewards = []
    state = env.reset()
    last_reward = None

    for i_episode in range(num_episodes):
        episode_reward = 0
        while True:
            action = epsilon_greedy_policy(
                state, epsilon, action_dim, device, Q)
            next_state, reward, done, _ = env.step(action)
            # print("payload 长度是{} 奖励{}".format(action, reward))
            episode_reward += reward
            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if done:
                state = env.reset()
                episode_rewards.append(episode_reward)
                if last_reward is None or episode_reward != last_reward:
                    last_reward = episode_reward
                else:
                    break

            if len(buffer) > batch_size:
                # 抽样
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = buffer.sample(
                    batch_size)
                state_batch = torch.tensor(
                    state_batch, dtype=torch.float32).to(device)
                action_batch = torch.tensor(
                    action_batch, dtype=torch.long).unsqueeze(1).repeat(1, 1, 1).to(device)
                reward_batch = torch.tensor(
                    reward_batch, dtype=torch.float32).unsqueeze(1).to(device)
                next_state_batch = torch.tensor(
                    next_state_batch, dtype=torch.float32).to(device)
                done_batch = torch.tensor(
                    done_batch, dtype=torch.float32).unsqueeze(1).to(device)

                # myations = Q(state_batch)
                # print(myations)
                # print(myations.shape)
                # print(action_batch.shape)
                # 计算Q值
                q_values = Q(state_batch).gather(2, action_batch)

                # 计算目标Q值
                q_values_next = Q_target(next_state_batch)
                q_values_next_max = q_values_next.max(dim=1, keepdim=True)[0]
                target_q_values = reward_batch + gamma * \
                    (1 - done_batch) * q_values_next_max

                # 计算损失并更新网络
                loss = F.mse_loss(q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新目标网络
                if i_episode % target_update == 0:
                    Q_target.load_state_dict(Q.state_dict())

        # 调整探索率
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 打印训练进度
        if (i_episode + 1) % 100 == 0:
            episode_range = i_episode + 1 - 100
            # print(episode_range)
            print("Episode {}/{}: Average reward = {:.2f} episode_range = {}".format(i_episode +
                  1, num_episodes, np.mean(episode_rewards[episode_range:]), episode_range))
    return Q


train(env_name='MyEnv',
      num_episodes=1000,
      batch_size=1,
      gamma=0.99,
      epsilon=1.0,
      epsilon_min=0.01,
      epsilon_decay=0.995,
      target_update=10,
      lr=0.001)
