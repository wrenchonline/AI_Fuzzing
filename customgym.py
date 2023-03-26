from simulation import emulate_program
import gym
from gym import spaces
import numpy as np
from queue import Queue
from threading import Thread
from utils import generate_payload_string
import pickle


from torchtext.data import Field
import torch


def process_disassembly(disassembly, vocab):
    # Define Field objects
    text_field = Field(tokenize=lambda x: x.split())

    # Preprocess strings
    preprocessed0 = text_field.preprocess(disassembly['op_str'])
    preprocessed1 = text_field.preprocess(disassembly['mnemonic'])

    # Convert strings to vocabulary indices
    indexed0 = [vocab.stoi[token] for token in preprocessed0]
    indexed1 = [vocab.stoi[token] for token in preprocessed1]

    # Convert to PyTorch tensors and concatenate them
    indexed0_tensor = torch.tensor(indexed0)
    indexed1_tensor = torch.tensor(indexed1)
    concatenated_tensor = torch.cat((indexed0_tensor, indexed1_tensor), dim=0)

    return concatenated_tensor


def combine_tensors(disassembly_tensor, return_address_tensor, is_call_tensor):
    # Add a new dimension to each tensor
    disassembly_tensor = disassembly_tensor
    return_address_tensor = return_address_tensor
    is_call_tensor = torch.unsqueeze(is_call_tensor, 0)  # 转换为形状为[1]的二维张量

    # 创建一个相同形状的全0张量
    combined_tensor = torch.zeros((3, 16), dtype=torch.uint8)

    # 将Disassembly tensor、Return Address tensor和isCall tensor分别拷贝到相应位置
    combined_tensor[0, :disassembly_tensor.shape[0]] = disassembly_tensor
    combined_tensor[1, :return_address_tensor.shape[0]] = return_address_tensor
    combined_tensor[2, :is_call_tensor.shape[0]] = is_call_tensor

    return combined_tensor


# 加载词汇表
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


# 创建一个空队列
q = Queue()


class MyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-1, high=255, shape=(3, 16))
        self.action_space = spaces.Discrete(12)
        self.seed()
        self.reset()
        self.last_action = None
        self.last_reward = None
        self.t = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.zeros((3, 16))
        self.last_action = None
        self.last_reward = None
        self.isreset = True
        self.record_return_addr = None
        self.iscall = False
        return self.state

    def step(self, action):
        global q
        done = False
        reward = 0
        assert self.action_space.contains(action), f"Invalid action {action}"
        if self.isreset:
            payload = generate_payload_string(1, action+1)
            self.t = Thread(target=emulate_program, args=(q, payload))
            self.isreset = False
            self.t.start()
        item = q.get()
        # 如果程序结束
        if item is None:
            done = True
            return torch.zeros((3, 16), dtype=torch.uint8), 0, done, {}
        # 处理元素
        disassembly = item["disassembly"]

        print(disassembly)
        # 将 disassembly 数组转换为 具有字典的词向量
        disassembly_tensor = process_disassembly(disassembly, vocab)

        #print("revice disassembly_tensor:%s\n" % item['return_adress'])

        return_address_arr = np.array(item['return_adress'])
        # 将 numpy 数组转换为 PyTorch Tensor
        return_address_tensor = torch.from_numpy(return_address_arr)
        # 将 isCall 值转换为 PyTorch Tensor，并将其作为一个标量值存储
        is_call_tensor = torch.tensor(
            item['isCall']).int().to(torch.uint8)

        # 关键如果call，我们记录返回地址
        if item['isCall']:
            self.iscall = item['isCall']
            self.record_return_addr = item['return_adress']
            reward = 0
            done = False
        elif self.record_return_addr:
            if self.record_return_addr == item['return_adress']:
                reward += 10
        self.state = combine_tensors(
            disassembly_tensor, return_address_tensor, is_call_tensor)

        if action == self.last_action:
            reward = self.last_reward

        # self.last_action = action
        # self.last_reward = reward
        info = {}
        return self.state, reward, done, info


y = MyEnv()

while True:
    state, reward, done, _ = y.step(10)
    if done:
        break
    print("reward %d\t\n" % reward)


payload = generate_payload_string(1, 2)
# gym.register(
#     id='MyEnv-v0',
#     entry_point='myenv:MyEnv',
# )
