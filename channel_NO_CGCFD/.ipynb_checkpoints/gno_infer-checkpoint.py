import os
import sys
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
sys.path.append('/home/sqj/LeetNeuralOperator/Transolver/PDE-Solving-StandardBenchmark')
from utils.normalizer import UnitTransformer

# 保持与训练完全一致的网络结构
def knn_graph(pos, k):
    dist = torch.cdist(pos, pos)
    _, idx = torch.topk(dist, k=k, dim=-1, largest=False)
    return idx

def gather_neighbors(x, idx):
    B, N, C = x.shape
    k = idx.shape[-1]
    batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
    return x[batch_idx, idx, :]

class GNO_Layer(nn.Module):
    def __init__(self, channels, k):
        super().__init__()
        self.k = k
        self.W = nn.Linear(channels, channels)
        self.kernel_mlp = nn.Sequential(nn.Linear(channels * 2 + 2, channels), nn.GELU(), nn.Linear(channels, channels))
        
    def forward(self, pos, h, idx):
        h_j = gather_neighbors(h, idx)
        pos_j = gather_neighbors(pos, idx)
        pos_diff = pos_j - pos.unsqueeze(2)
        h_i = h.unsqueeze(2).expand_as(h_j)
        msg_in = torch.cat([h_i, h_j, pos_diff], dim=-1)
        msg = self.kernel_mlp(msg_in).mean(dim=2)
        return F.gelu(self.W(h) + msg)

class GNO(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=3, layers=4, k=10):
        super().__init__()
        self.k = k
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.gno_layers = nn.ModuleList([GNO_Layer(hidden_channels, k) for _ in range(layers)])
        self.fc_out = nn.Sequential(nn.Linear(hidden_channels, 128), nn.GELU(), nn.Linear(128, out_channels))

    def forward(self, pos, fx):
        x = torch.cat([pos, fx], dim=-1)
        h = self.fc_in(x)
        idx = knn_graph(pos, self.k)
        for layer in self.gno_layers:
            h = layer(pos, h, idx)
        return self.fc_out(h)

def main():
    parser = argparse.ArgumentParser('GNO Zero-Shot Inference')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/channel_GNO.pt')
    parser.add_argument('--data_path', type=str, default='./channel_data')
    parser.add_argument('--nx', type=int, default=100)
    parser.add_argument('--ny', type=int, default=15)
    parser.add_argument('--u_in', type=float, default=5.0)
    parser.add_argument('--nu', type=float, default=0.05)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_data = np.load(f"{args.data_path}/Channel_X.npy")
    N = full_data.shape[0]
    ntrain = 1 if N == 1 else int(0.8 * N)

    inputX = torch.tensor(full_data[:ntrain], dtype=torch.float)
    inputY = torch.tensor(np.load(f"{args.data_path}/Channel_Y.npy")[:ntrain], dtype=torch.float)
    input_orig = torch.stack([inputX, inputY], dim=-1).reshape(ntrain, -1, 2)
    
    output_orig = torch.tensor(np.load(f"{args.data_path}/Channel_Q.npy")[:ntrain], dtype=torch.float).permute(0, 2, 3, 1).reshape(ntrain, -1, 3)
    
    params_orig = torch.tensor(np.load(f"{args.data_path}/Channel_Params.npy")[:ntrain], dtype=torch.float).unsqueeze(1).repeat(1, input_orig.shape[1], 1)

    x_normalizer = UnitTransformer(input_orig).to(device)
    fx_normalizer = UnitTransformer(params_orig).to(device)
    y_normalizer = UnitTransformer(output_orig).to(device)

    x_grid = np.linspace(0, 10, args.nx)
    y_grid = np.linspace(0, 1, args.ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    x_infer = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32).reshape(1, -1, 2).to(device)
    fx_infer = torch.tensor([args.u_in, args.nu], dtype=torch.float32).view(1, 1, 2).repeat(1, args.nx * args.ny, 1).to(device)

    x_infer_encoded = x_normalizer.encode(x_infer)
    fx_infer_encoded = fx_normalizer.encode(fx_infer)

    model = GNO(in_channels=4, hidden_channels=64, out_channels=3, layers=4, k=10).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        out_encoded = model(x_infer_encoded, fx_infer_encoded)
        out = y_normalizer.decode(out_encoded)
    
    pred_u = out[0, :, 0].reshape(args.nx, args.ny).cpu().numpy()
    pred_v = out[0, :, 1].reshape(args.nx, args.ny).cpu().numpy()
    pred_p = out[0, :, 2].reshape(args.nx, args.ny).cpu().numpy()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    fig.suptitle(f"GNO Inference | Grid: {args.nx}x{args.ny}, $U_{{in}}$={args.u_in}, $\\nu$={args.nu}", fontsize=14)
    axes[0].pcolormesh(X, Y, pred_u, shading='auto', cmap='jet')
    axes[1].pcolormesh(X, Y, pred_v, shading='auto', cmap='jet')
    axes[2].pcolormesh(X, Y, pred_p, shading='auto', cmap='jet')
    plt.tight_layout()
    plt.savefig(f"gno_infer_U{args.u_in}_nu{args.nu}_grid{args.nx}x{args.ny}.png", dpi=150)
    print(f"GNO 推理完成！图像保存为: gno_infer...")

if __name__ == '__main__':
    main()