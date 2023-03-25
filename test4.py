import json

import torchtext.vocab as vocab
from torchtext.data import Field
import torch


# 从文件中读取 JSON 数据
with open('bin_array.json', 'r') as f:
    datas = json.load(f)

# 打印读取到的 JSON 数据
print(datas)

myvocab = []
for dat in datas:
    myvocab.append(dat["mnemonic"])
    myvocab.append(dat["op_str"])

myvocabs = " ".join(myvocab)
# 定义Field对象
text_field = Field(tokenize=lambda x: x.split())
# 预处理字符串
preprocessed = text_field.preprocess(myvocabs)


# 构建词汇表
text_field.build_vocab([preprocessed], vectors=vocab.GloVe(name='6B', dim=100))

# 查看词汇表大小
vocab_size = len(text_field.vocab)
print('Vocabulary size:', vocab_size)

# 查看词汇表中的单词
print('Vocabulary words:', text_field.vocab.itos)


# 将字符串转换为词汇表索引
indexed = [text_field.vocab.stoi[token] for token in preprocessed]


# 将索引转换为张量
tensor = torch.Tensor(indexed)

# 打印张量
print(tensor)
