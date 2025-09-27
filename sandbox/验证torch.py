# 验证是否成功安装了 PyTorch
import torch

print(torch.__version__)  # 打印 PyTorch 版本
print(torch.cuda.is_available())  # 检查是否可以使用 GPU
