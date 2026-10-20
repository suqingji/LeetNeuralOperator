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
# 1. 定义 GNO 算子
# ==========================================
def knn_graph(pos, k):
    """基于空间坐标动态构建 K 近邻图 (B, N, k)"""
    dist = torch.cdist(pos, pos)
    _, idx = torch.topk(dist, k=k, dim=-1, largest=False)
    return idx

def gather_neighbors(x, idx):
    """收集邻居节点的特征"""
    B, N, C = x.shape
    k = idx.shape[-1]
    batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
    return x[batch_idx, idx, :]

class GNO_Layer(nn.Module):
    def __init__(self, channels, k):
        super().__init__()
        self.k = k
        self.W = nn.Linear(channels, channels)
        # 核函数近似 MLP：输入为节点i特征、节点j特征以及相对坐标
        self.kernel_mlp = nn.Sequential(
            nn.Linear(channels * 2 + 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        
    def forward(self, pos, h, idx):
        h_j = gather_neighbors(h, idx)
        pos_j = gather_neighbors(pos, idx)
        
        pos_diff = pos_j - pos.unsqueeze(2)
        h_i = h.unsqueeze(2).expand_as(h_j)
        
        msg_in = torch.cat([h_i, h_j, pos_diff], dim=-1)
        msg = self.kernel_mlp(msg_in).mean(dim=2) # 聚合邻居信息 (积分算子的蒙特卡洛/求和近似)
        
        return F.gelu(self.W(h) + msg)

class GNO(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=3, layers=4, k=10):
        super().__init__()
        self.k = k
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.gno_layers = nn.ModuleList([GNO_Layer(hidden_channels, k) for _ in range(layers)])
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.GELU(),
            nn.Linear(128, out_channels)
        )

    def forward(self, pos, fx):
        # 拼接物理参数特征 (Batch, N, 2) + 坐标 (Batch, N, 2) -> (Batch, N, 4)
        x = torch.cat([pos, fx], dim=-1)
        h = self.fc_in(x)
        idx = knn_graph(pos, self.k) # 每层复用同一个空间拓扑图
        
        for layer in self.gno_layers:
            h = layer(pos, h, idx)
            
        return self.fc_out(h)

# ==========================================
# 2. 训练主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser('Training GNO for Channel Flow')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=4) # 图卷积显存占用稍大，减小batch
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument('--save_name', type=str, default='channel_GNO')
    parser.add_argument('--data_path', type=str, default='./channel_data')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    inputX = torch.tensor(np.load(f"{args.data_path}/Channel_X.npy"), dtype=torch.float)
    inputY = torch.tensor(np.load(f"{args.data_path}/Channel_Y.npy"), dtype=torch.float)
    output = torch.tensor(np.load(f"{args.data_path}/Channel_Q.npy"), dtype=torch.float)
    output = output.permute(0, 2, 3, 1)
    
    params = torch.tensor(np.load(f"{args.data_path}/Channel_Params.npy"), dtype=torch.float)

    N = inputX.shape[0]
    ntrain = 1 if N == 1 else int(0.8 * N)
    ntest = 1 if N == 1 else N - ntrain

    x_data = torch.stack([inputX, inputY], dim=-1).reshape(N, -1, 2)
    y_data = output.reshape(N, -1, 3)
    
    # 将标量参数扩展至每个节点 (Batch, N_pts, 2)
    fx_data = params.unsqueeze(1).repeat(1, x_data.shape[1], 1)

    x_train, x_test = x_data[:ntrain], x_data[-ntest:]
    fx_train, fx_test = fx_data[:ntrain], fx_data[-ntest:]
    y_train, y_test = y_data[:ntrain], y_data[-ntest:]

    #x_normalizer = UnitTransformer(x_train).cuda()
    #fx_normalizer = UnitTransformer(fx_train).cuda()
    #y_normalizer = UnitTransformer(y_train).cuda()
 
    #x_train, x_test = x_normalizer.encode(x_train.cuda()), x_normalizer.encode(x_test.cuda())
    #fx_train, fx_test = fx_normalizer.encode(fx_train.cuda()), fx_normalizer.encode(fx_test.cuda())
    #y_train = y_normalizer.encode(y_train.cuda())
    #y_test = y_test.cuda() # 评估时使用
    
    # 1. 实例化 Normalizer（此时在 CPU 上）
    x_normalizer = UnitTransformer(x_train)
    fx_normalizer = UnitTransformer(fx_train)
    y_normalizer = UnitTransformer(y_train)

    # 2. 统一在 CPU 上进行归一化编码
    x_train, x_test = x_normalizer.encode(x_train), x_normalizer.encode(x_test)
    fx_train, fx_test = fx_normalizer.encode(fx_train), fx_normalizer.encode(fx_test)
    y_train = y_normalizer.encode(y_train)

    # 3. 将 Normalizer 的内部参数移动到 GPU，以备后续计算 loss 解码时使用
    x_normalizer.cuda()
    fx_normalizer.cuda()
    y_normalizer.cuda()

    # 构建 DataLoader（数据保持在内存中）
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, fx_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, fx_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = GNO(in_channels=4, hidden_channels=64, out_channels=3, layers=4, k=10).cuda()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GNO Total Trainable Params: {total_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    for ep in range(args.epochs):
        model.train()
        train_loss = 0
        for x, fx, y in train_loader:
            # 4. 在每一次 Batch 迭代时，将当前的数据块送入 GPU
            x, fx, y = x.cuda(), fx.cuda(), y.cuda()
            
            optimizer.zero_grad()
            out = model(x, fx)
            loss = myloss(y_normalizer.decode(out), y_normalizer.decode(y))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()

        if ep % 10 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for x, fx, y in test_loader:
                    # 测试集迭代同样需要放入 GPU
                    x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                    
                    out = model(x, fx)
                    test_loss += myloss(y_normalizer.decode(out), y).item()
            print(f"Epoch {ep:4d} | Train loss: {train_loss / ntrain:.5f} | Test loss: {test_loss / ntest:.5f}")

    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('./checkpoints', args.save_name + '.pt'))
    print(f"训练完成！模型已保存至 ./checkpoints/{args.save_name}.pt")

if __name__ == "__main__":
    main()