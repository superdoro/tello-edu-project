import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

# 在這隻乾淨的程式裡下載，絕對不會遇到 WinError 6
# weights = Raft_Large_Weights.DEFAULT
# model = raft_large(weights=weights, progress=True)

weights = Raft_Small_Weights.DEFAULT
model = raft_small(weights=weights, progress=True)