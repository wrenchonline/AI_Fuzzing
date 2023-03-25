import torchtext.vocab as vocab
from torchtext.data import Field
import torch
# 定义Field对象
text_field = Field(tokenize=lambda x: x.split())

# 要转换的字符串
s = "byte ptr [rdi], byte ptr [rsi]"

# 预处理字符串
preprocessed = text_field.preprocess(s)

letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
myletters = " ".join(letters)


# 预处理字符串
premyletters = text_field.preprocess(myletters)


# 构建词汇表
text_field.build_vocab([preprocessed], vectors=vocab.GloVe(name='6B', dim=100))

# 查看词汇表大小
vocab_size = len(text_field.vocab)
print('Vocabulary size:', vocab_size)

# 查看词汇表中的单词
print('Vocabulary words:', text_field.vocab.itos)


# 将字符串转换为词汇表索引
indexed = [text_field.vocab.stoi[token] for token in premyletters]

# 将索引转换为张量
tensor = torch.Tensor(indexed)

# 打印张量
print(tensor)
