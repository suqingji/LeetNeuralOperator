import os
import sys
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 屏蔽 timm 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

# 引入之前的 Loss 与 Normalizer
sys.path.append('/home/sqj/LeetNeuralOperator/Transolver/PDE-Solving-StandardBenchmark')
from utils.testloss import TestLoss
from utils.normalizer import UnitTransformer

# ==========================================
# 1. 定义标准的 2D 傅里叶神经算子 (FNO2d)
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
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # 计算 2D 傅里叶变换
        x_ft = torch.fft.rfft2(x)

        # 截断高频，仅保留低频 mode 并进行线性变换
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 逆傅里叶变换恢复到物理空间
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # 填充以防网格边缘截断
        
        # 输入通道为 4 (X, Y, U_in, nu)
        self.fc0 = nn.Linear(4, self.width)
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3) # 输出 u, v, p

    def forward(self, x):
        # FNO的输入要求保持网格维度: (Batch, H, W, Channels)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # 转为 (Batch, Channels, H, W) 供卷积层使用
        
        # Fourier 层 + 残差连接
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
        
        x = x.permute(0, 2, 3, 1) # 变回 (Batch, H, W, Channels)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

# ==========================================
# 2. 训练主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser('Training FNO for Channel Flow')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument('--save_name', type=str, default='channel_FNO')
    parser.add_argument('--data_path', type=str, default='./channel_data')
    
    # FNO 专用超参数
    parser.add_argument('--modes1', type=int, default=12, help='X方向保留的傅里叶模态数')
    parser.add_argument('--modes2', type=int, default=12, help='Y方向保留的傅里叶模态数')
    parser.add_argument('--width', type=int, default=32, help='FNO隐藏通道数')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. 加载数据
    inputX = torch.tensor(np.load(f"{args.data_path}/Channel_X.npy"), dtype=torch.float)
    inputY = torch.tensor(np.load(f"{args.data_path}/Channel_Y.npy"), dtype=torch.float)
    
    # 【修复核心】：动态获取 Batch 数和当前网格的 H, W 尺寸
    Batch, H, W = inputX.shape 
    
    output = torch.tensor(np.load(f"{args.data_path}/Channel_Q.npy"), dtype=torch.float)
    output = output.permute(0, 2, 3, 1) # 动态自适应 (Batch, H, W, 3)

    params = torch.tensor(np.load(f"{args.data_path}/Channel_Params.npy"), dtype=torch.float)
    # 将标量参数动态扩展至全场网格节点 (Batch, H, W, 2)
    params = params.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)

    # 在通道维度上将 坐标 与 物理参数 拼接，作为 FNO 的统一输入 (Batch, H, W, 4)
    in_data = torch.cat([torch.stack([inputX, inputY], dim=-1), params], dim=-1)

    N = in_data.shape[0]
    if N == 1:
        ntrain, ntest = 1, 1
    else:
        ntrain = int(0.8 * N)
        ntest = N - ntrain

    # FNO 需要保持 2D 网格结构，但 Normalizer 接收铺平的维度，所以需要先 Flatten 后 Reshape
    x_train_flat = in_data[:ntrain].reshape(ntrain, -1, 4)
    x_test_flat = in_data[-ntest:].reshape(ntest, -1, 4)
    y_train_flat = output[:ntrain].reshape(ntrain, -1, 3)
    y_test_flat = output[-ntest:].reshape(ntest, -1, 3)

    # 实例化归一化器
    x_normalizer = UnitTransformer(x_train_flat)
    y_normalizer = UnitTransformer(y_train_flat)

    # Encode 并还原为网格维度供 FNO 使用
    x_train = x_normalizer.encode(x_train_flat).reshape(ntrain, H, W, 4)
    x_test = x_normalizer.encode(x_test_flat).reshape(ntest, H, W, 4)
    y_train = y_normalizer.encode(y_train_flat).reshape(ntrain, H, W, 3)
    y_test = y_normalizer.encode(y_test_flat).reshape(ntest, H, W, 3)

    # GPU
    x_normalizer.cuda()
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    # 2. 构建模型
    model = FNO2d(modes1=args.modes1, modes2=args.modes2, width=args.width).cuda()
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FNO Total Trainable Params: {total_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    # 3. 训练循环
    for ep in range(args.epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            
            # 模型预测 (Batch, H, W, 3)
            out = model(x)
            
            # 为了计算 Loss 和反归一化，临时展平
            out_flat = out.reshape(x.shape[0], -1, 3)
            y_flat = y.reshape(x.shape[0], -1, 3)
            
            out_decode = y_normalizer.decode(out_flat)
            y_decode = y_normalizer.decode(y_flat)

            loss = myloss(out_decode, y_decode)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()

        if ep % 10 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()
                    out = model(x)
                    out_flat = out.reshape(x.shape[0], -1, 3)
                    y_flat = y.reshape(x.shape[0], -1, 3)
                    test_loss += myloss(y_normalizer.decode(out_flat), y_normalizer.decode(y_flat)).item()
            
            print(f"Epoch {ep:4d} | Train loss: {train_loss / ntrain:.5f} | Test loss: {test_loss / ntest:.5f}")

    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('./checkpoints', args.save_name + '.pt'))
    print(f"训练完成！模型已保存至 ./checkpoints/{args.save_name}.pt")

if __name__ == "__main__":
    main()