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

# ==========================================
# 1. 必须包含与训练时完全相同的 FNO 网络结构
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    #def forward(self, x):
    #    batchsize = x.shape[0]
    #    x_ft = torch.fft.rfft2(x)
    #    out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
    #    out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
    #    out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
    #    x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
    #    return x
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        # 动态计算当前网格能够支持的实际最大模态数
        m1 = min(self.modes1, x_ft.shape[-2])
        m2 = min(self.modes2, x_ft.shape[-1])
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # 动态切片输入频域和权重矩阵
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], 
            self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], 
            self.weights2[:, :, :m1, :m2]
        )

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.fc0 = nn.Linear(4, self.width) # 4 个通道输入
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3) 

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

def main():
    parser = argparse.ArgumentParser('FNO Zero-Shot Inference')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/channel_FNO.pt')
    parser.add_argument('--data_path', type=str, default='./channel_data')
    
    # 物理参数推断设置
    parser.add_argument('--nx', type=int, default=100, help='X方向网格节点数')
    parser.add_argument('--ny', type=int, default=10, help='Y方向网格节点数')
    parser.add_argument('--u_in', type=float, default=5.0, help='入口速度参数')
    parser.add_argument('--nu', type=float, default=0.05, help='运动粘度参数')
    
    # FNO 结构参数
    parser.add_argument('--modes1', type=int, default=12)
    parser.add_argument('--modes2', type=int, default=12)
    parser.add_argument('--width', type=int, default=32)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 动态恢复 Normalizer 的缩放比例 (将 X, Y 和 Params 在通道维度上拼接以符合训练时逻辑)
    full_data = np.load(f"{args.data_path}/Channel_X.npy")
    N = full_data.shape[0]
    ntrain = 1 if N == 1 else int(0.8 * N)

    inputX = torch.tensor(full_data[:ntrain], dtype=torch.float)
    inputY = torch.tensor(np.load(f"{args.data_path}/Channel_Y.npy")[:ntrain], dtype=torch.float)
    
    # 【修复核心】：动态获取当前生成的原始数据 H 和 W
    _, H, W = inputX.shape
    
    output_orig = torch.tensor(np.load(f"{args.data_path}/Channel_Q.npy")[:ntrain], dtype=torch.float)
    output_orig = output_orig.permute(0, 2, 3, 1).reshape(ntrain, -1, 3)
    
    params_orig = torch.tensor(np.load(f"{args.data_path}/Channel_Params.npy")[:ntrain], dtype=torch.float)
    # 使用动态获取的 H 和 W 替换掉原本写死的 200 和 30
    params_orig = params_orig.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)

    in_orig = torch.cat([torch.stack([inputX, inputY], dim=-1), params_orig], dim=-1).reshape(ntrain, -1, 4)

    x_normalizer = UnitTransformer(in_orig).to(device)
    y_normalizer = UnitTransformer(output_orig).to(device)

    # 3. 生成推理用的自定义 2D 网格与物理参数特征
    x_grid = np.linspace(0, 10, args.nx)
    y_grid = np.linspace(0, 1, args.ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # 组装特征 (1, nx, ny, 4)
    x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)
    u_tensor = torch.full((args.nx, args.ny, 1), args.u_in, dtype=torch.float32)
    nu_tensor = torch.full((args.nx, args.ny, 1), args.nu, dtype=torch.float32)
    
    infer_input = torch.cat([x_tensor, y_tensor, u_tensor, nu_tensor], dim=-1).unsqueeze(0).to(device)
    
    # 将其展平进行归一化编码后，再恢复为 2D 形状给 FNO 用
    infer_input_flat = infer_input.reshape(1, -1, 4)
    infer_input_encoded = x_normalizer.encode(infer_input_flat).reshape(1, args.nx, args.ny, 4)

    # 4. 加载模型
    model = FNO2d(modes1=args.modes1, modes2=args.modes2, width=args.width).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    # 5. 推理执行
    with torch.no_grad():
        out_encoded = model(infer_input_encoded)
        # 展平以便通过反归一化，然后恢复形状
        out_flat = out_encoded.reshape(1, -1, 3)
        out = y_normalizer.decode(out_flat).reshape(args.nx, args.ny, 3)
    
    pred_u = out[:, :, 0].cpu().numpy()
    pred_v = out[:, :, 1].cpu().numpy()
    pred_p = out[:, :, 2].cpu().numpy()

    # 6. 画图渲染
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    fig.suptitle(f"FNO Inference | Grid: {args.nx}x{args.ny}, $U_{{in}}$={args.u_in}, $\\nu$={args.nu}", fontsize=14)

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
    save_path = f"fno_infer_U{args.u_in}_nu{args.nu}_grid{args.nx}x{args.ny}.png"
    plt.savefig(save_path, dpi=150)
    print(f"FNO 推理完成！图像保存为: {save_path}")

if __name__ == '__main__':
    main()