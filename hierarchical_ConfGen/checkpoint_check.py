import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cpu')
checkpoint_path = "/data/esmdiff_ckpt/epoch_014.ckpt/checkpoint/mp_rank_00_model_states.pt"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# 키 목록 확인
print("✅ Checkpoint Keys:")
for key in checkpoint.keys():
    print(f"{key}")
print(checkpoint["param_shapes"])