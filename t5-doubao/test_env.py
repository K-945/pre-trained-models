# 测试Python环境
import sys
print("Python版本:", sys.version)

try:
    import torch
    print("PyTorch版本:", torch.__version__)
    print("CUDA可用:", torch.cuda.is_available())
    print("设备:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
except Exception as e:
    print("导入PyTorch失败:", e)

try:
    import pandas as pd
    print("Pandas版本:", pd.__version__)
except Exception as e:
    print("导入Pandas失败:", e)

print("\n环境测试完成!")
