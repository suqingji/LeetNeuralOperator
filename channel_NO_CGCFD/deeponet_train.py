import os
import sys
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
sys.path.append('/home/sqj/LeetNeuralOperator/Transolver/PDE-Solving-StandardBenchmark')
from utils.testloss import TestLoss
from utils.normalizer import UnitTransformer

# ==========================================
# 1. 定义 DeepONet 模型
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
        
        # 分支网络：输入物理参数 (Batch, 2)
        self.branch = MLP(in_dim=branch_dim, hidden_dim=hidden_dim, out_dim=p * out_channels, layers=3)
        # 主干网络：输入空间坐标 (Batch, N_pts, 2)
        self.trunk = MLP(in_dim=trunk_dim, hidden_dim=hidden_dim, out_dim=p * out_channels, layers=3)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, branch_in, trunk_in):
        # B shape: (Batch, p * 3) -> (Batch, 1, 3, p)
        B = self.branch(branch_in).view(branch_in.shape[0], 1, self.out_channels, self.p)
        # T shape: (Batch, N_pts, p * 3) -> (Batch, N_pts, 3, p)
        T = self.trunk(trunk_in).view(trunk_in.shape[0], trunk_in.shape[1], self.out_channels, self.p)
        
        # 在特征维度 p 上进行点积求和，加上偏置
        out = torch.sum(B * T, dim=-1) + self.bias # (Batch, N_pts, 3)
        return out

# ==========================================
# 2. 训练主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser('Training DeepONet for Channel Flow')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument('--save_name', type=str, default='channel_DeepONet')
    parser.add_argument('--data_path', type=str, default='./channel_data')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 加载原始数据
    inputX = torch.tensor(np.load(f"{args.data_path}/Channel_X.npy"), dtype=torch.float)
    inputY = torch.tensor(np.load(f"{args.data_path}/Channel_Y.npy"), dtype=torch.float)
    output = torch.tensor(np.load(f"{args.data_path}/Channel_Q.npy"), dtype=torch.float)
    output = output.permute(0, 2, 3, 1) # (Batch, 200, 30, 3)
    
    # DeepONet 的 Branch 输入是全局标量参数 (Batch, 2)
    params = torch.tensor(np.load(f"{args.data_path}/Channel_Params.npy"), dtype=torch.float)

    N = inputX.shape[0]
    ntrain = 1 if N == 1 else int(0.8 * N)
    ntest = 1 if N == 1 else N - ntrain

    # 将 2D 网格展平为点云序列，因为 Trunk Net 按点处理
    x_data = torch.stack([inputX, inputY], dim=-1).reshape(N, -1, 2)
    y_data = output.reshape(N, -1, 3)

    x_train, x_test = x_data[:ntrain], x_data[-ntest:]
    fx_train, fx_test = params[:ntrain], params[-ntest:]
    y_train, y_test = y_data[:ntrain], y_data[-ntest:]

    x_normalizer = UnitTransformer(x_train)
    fx_normalizer = UnitTransformer(fx_train)
    y_normalizer = UnitTransformer(y_train)

    x_train, x_test = x_normalizer.encode(x_train), x_normalizer.encode(x_test)
    fx_train, fx_test = fx_normalizer.encode(fx_train), fx_normalizer.encode(fx_test)
    y_train = y_normalizer.encode(y_train)

    x_normalizer.cuda()
    fx_normalizer.cuda()
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, fx_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, fx_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = DeepONet(branch_dim=2, trunk_dim=2, hidden_dim=128, p=128, out_channels=3).cuda()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DeepONet Total Trainable Params: {total_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    for ep in range(args.epochs):
        model.train()
        train_loss = 0
        for x, fx, y in train_loader:
            x, fx, y = x.cuda(), fx.cuda(), y.cuda()
            optimizer.zero_grad()
            
            out = model(fx, x) # Branch(fx) * Trunk(x)
            
            out_decode = y_normalizer.decode(out)
            y_decode = y_normalizer.decode(y)

            loss = myloss(out_decode, y_decode)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()

        if ep % 10 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for x, fx, y in test_loader:
                    x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                    out = model(fx, x)
                    test_loss += myloss(y_normalizer.decode(out), y_normalizer.decode(y)).item()
            print(f"Epoch {ep:4d} | Train loss: {train_loss / ntrain:.5f} | Test loss: {test_loss / ntest:.5f}")

    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('./checkpoints', args.save_name + '.pt'))
    print(f"训练完成！模型已保存至 ./checkpoints/{args.save_name}.pt")

if __name__ == "__main__":
    main()