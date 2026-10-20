import os
import sys
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
sys.path.append('/home/sqj/LeetNeuralOperator/Transolver/PDE-Solving-StandardBenchmark')
from utils.normalizer import UnitTransformer

# ==========================================
# 1. 包含与训练时相同的网络结构
# ==========================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        super().__init__()
        modules = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        modules.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, branch_dim=2, trunk_dim=2, hidden_dim=128, p=128, out_channels=3):
        super().__init__()
        self.p = p
        self.out_channels = out_channels
        self.branch = MLP(in_dim=branch_dim, hidden_dim=hidden_dim, out_dim=p * out_channels, layers=3)
        self.trunk = MLP(in_dim=trunk_dim, hidden_dim=hidden_dim, out_dim=p * out_channels, layers=3)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, branch_in, trunk_in):
        B = self.branch(branch_in).view(branch_in.shape[0], 1, self.out_channels, self.p)
        T = self.trunk(trunk_in).view(trunk_in.shape[0], trunk_in.shape[1], self.out_channels, self.p)
        out = torch.sum(B * T, dim=-1) + self.bias 
        return out

def main():
    parser = argparse.ArgumentParser('DeepONet Zero-Shot Inference')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/channel_DeepONet.pt')
    parser.add_argument('--data_path', type=str, default='./channel_data')
    parser.add_argument('--nx', type=int, default=100)
    parser.add_argument('--ny', type=int, default=15)
    parser.add_argument('--u_in', type=float, default=5.0)
    parser.add_argument('--nu', type=float, default=0.05)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 动态匹配重建数据的形状，恢复归一化器
    full_data = np.load(f"{args.data_path}/Channel_X.npy")
    N = full_data.shape[0]
    ntrain = 1 if N == 1 else int(0.8 * N)

    inputX = torch.tensor(full_data[:ntrain], dtype=torch.float)
    inputY = torch.tensor(np.load(f"{args.data_path}/Channel_Y.npy")[:ntrain], dtype=torch.float)
    input_orig = torch.stack([inputX, inputY], dim=-1).reshape(ntrain, -1, 2)
    
    output_orig = torch.tensor(np.load(f"{args.data_path}/Channel_Q.npy")[:ntrain], dtype=torch.float)
    output_orig = output_orig.permute(0, 2, 3, 1).reshape(ntrain, -1, 3)
    
    # 注意：DeepONet 的 Param 特征无需按点展开
    params_orig = torch.tensor(np.load(f"{args.data_path}/Channel_Params.npy")[:ntrain], dtype=torch.float)

    x_normalizer = UnitTransformer(input_orig).to(device)
    fx_normalizer = UnitTransformer(params_orig).to(device)
    y_normalizer = UnitTransformer(output_orig).to(device)

    # 构建任意密度的推理网格，将其展平作为 Trunk 网络的输入
    x_grid = np.linspace(0, 10, args.nx)
    y_grid = np.linspace(0, 1, args.ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    x_infer = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32).reshape(1, -1, 2).to(device)
    # 将物理参数作为 Branch 网络的输入
    fx_infer = torch.tensor([[args.u_in, args.nu]], dtype=torch.float32).to(device)

    x_infer_encoded = x_normalizer.encode(x_infer)
    fx_infer_encoded = fx_normalizer.encode(fx_infer)

    model = DeepONet(branch_dim=2, trunk_dim=2, hidden_dim=128, p=128, out_channels=3).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        out_encoded = model(fx_infer_encoded, x_infer_encoded)
        out = y_normalizer.decode(out_encoded)
    
    # (1, nx*ny, 3) 还原成 (nx, ny) 2D矩阵
    pred_u = out[0, :, 0].reshape(args.nx, args.ny).cpu().numpy()
    pred_v = out[0, :, 1].reshape(args.nx, args.ny).cpu().numpy()
    pred_p = out[0, :, 2].reshape(args.nx, args.ny).cpu().numpy()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    fig.suptitle(f"DeepONet Inference | Grid: {args.nx}x{args.ny}, $U_{{in}}$={args.u_in}, $\\nu$={args.nu}", fontsize=14)

    c1 = axes[0].pcolormesh(X, Y, pred_u, shading='auto', cmap='jet')
    axes[0].set_title('Predicted Velocity U')
    fig.colorbar(c1, ax=axes[0])

    c2 = axes[1].pcolormesh(X, Y, pred_v, shading='auto', cmap='jet')
    axes[1].set_title('Predicted Velocity V')
    fig.colorbar(c2, ax=axes[1])

    c3 = axes[2].pcolormesh(X, Y, pred_p, shading='auto', cmap='jet')
    axes[2].set_title('Predicted Pressure P')
    fig.colorbar(c3, ax=axes[2])

    plt.tight_layout()
    save_path = f"deeponet_infer_U{args.u_in}_nu{args.nu}_grid{args.nx}x{args.ny}.png"
    plt.savefig(save_path, dpi=150)
    print(f"DeepONet 推理完成！图像保存为: {save_path}")

if __name__ == '__main__':
    main()