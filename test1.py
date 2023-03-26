import torch

disassembly_tensor = torch.tensor([2, 3, 0, 0, 22])
return_address_tensor = torch.zeros(8, dtype=torch.uint8)
is_call_tensor = torch.tensor([1], dtype=torch.uint8)

result_tensor = torch.cat((disassembly_tensor.unsqueeze(
    0), return_address_tensor.unsqueeze(0)), dim=0)
result_tensor = torch.cat((result_tensor, is_call_tensor.unsqueeze(0)), dim=0)

print(result_tensor)
