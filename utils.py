import random
import string
from gym import spaces


def generate_payload_string(min_length, max_length):
    # 生成一个随机长度的字符串
    length = random.randint(min_length, max_length)
    # 从大写字母、小写字母和数字中随机选择字符
    characters = string.ascii_letters + string.digits
    # 生成随机字符串
    payload = ''.join(random.choice(characters) for _ in range(length))
    return payload


# payload = generate_payload_string(1, 12)
# print(payload)


# action_space = spaces.Discrete(12)

# print(action_space.sample())
