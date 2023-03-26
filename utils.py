import random
import string
from gym import spaces


def generate_payload_string(length):
    # 从大写字母、小写字母和数字中随机选择字符
    characters = string.ascii_letters + string.digits
    # 生成随机字符串
    payload = ''.join(random.choice(characters) for _ in range(length))
    return payload
