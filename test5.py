import torch
import numpy as np
import pickle
from torchtext.data import Field


# 加载词汇表
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


disassembly = {'address': 174354190, 'mnemonic': 'add',
               'op_str': 'byte ptr [rax], al', 'return_adress': bytearray(b'\x00\x00\x00\x00\x00\x00\x00\x00'), 'isCall': 1}


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


disassembly_tensor = process_disassembly(disassembly, vocab)
return_address_arr = np.array(disassembly['return_adress'])
# 将 numpy 数组转换为 PyTorch Tensor
return_address_tensor = torch.from_numpy(return_address_arr)
# 将 isCall 值转换为 PyTorch Tensor，并将其作为一个标量值存储
is_call_tensor = torch.tensor(disassembly['isCall']).int().to(torch.uint8)


# 打印所有 Tensor
print('Disassembly Tensor:', disassembly_tensor)
print('Return Address Tensor:', return_address_tensor)
print('isCall Tensor:', is_call_tensor)

# aaa = torch.tensor([[2,  3,  0,  0, 22],
#                     [0,  0,  0,  0,  0,  0,  0,  0],
#                     [1]])

#print('combine_tensors:', is_call_tensor)


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


combined_tensor = combine_tensors(
    disassembly_tensor, return_address_tensor, is_call_tensor)

print(combined_tensor.shape)
print(combined_tensor)
