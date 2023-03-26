import torch

# 创建一个包含 Q 值的张量，形状为 [1, 3, 20]
Q = torch.randn(1, 3, 20)

# 定义一个需要提取 Q 值的动作
action = 19

# 使用 torch.gather() 函数提取指定动作对应的 Q 值
Q_action = torch.gather(
    Q, dim=2, index=torch.tensor([[action]]).repeat(1, 3, 1))

# 打印输出结果
print(Q_action)
