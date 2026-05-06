import sys, platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print(f'Python    : {sys.version[:6]}')
print(f'NumPy     : {np.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'OS        : {platform.system()} {platform.release()}')

# 아래 코드는 Week 7부터 PyTorch 설치 후 사용
try:
    import torch
    print(f'PyTorch   : {torch.__version__}')
    print(f'CUDA      : {torch.cuda.is_available()}')
except ImportError:
    print('PyTorch : not installed -- will be added in Week 7')