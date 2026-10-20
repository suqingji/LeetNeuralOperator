import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

# 配置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 定义 PINN 模型
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)  # 输出 u, v, p
        )
        
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

# 2. 偏微分方程 (Navier-Stokes) 损失计算
def pde_loss(model, x, y, nu):
    out = model(x, y)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # 计算一阶导数
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]

    # 计算二阶导数
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]

    # 连续性方程
    eq_continuity = u_x + v_y
    # X 动量方程
    eq_momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    # Y 动量方程
    eq_momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return torch.mean(eq_continuity**2) + torch.mean(eq_momentum_x**2) + torch.mean(eq_momentum_y**2)

# 3. 数据生成主流程
def generate_data():
    nx, ny = 200, 30
    x_grid = np.linspace(0, 10, nx)
    y_grid = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # 转换为预测用的 Tensor
    X_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device)
    Y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32, device=device)

    # 设定参数范围：对数分布 10 个样本
    U_list = np.logspace(np.log10(10), np.log10(10), 1)
    nu_list = np.logspace(np.log10(0.01), np.log10(0.1), 5)  # 设定粘性系数范围

    # 存储数组初始化
    all_X = np.zeros((100, nx, ny))
    all_Y = np.zeros((100, nx, ny))
    all_Q = np.zeros((100, 3, nx, ny))  # 通道为3：u, v, p
    all_Params = np.zeros((100, 2))     # 存储对应的 U 和 nu

    sample_idx = 0
    total_samples = len(U_list) * len(nu_list)

    for u_in in U_list:
        for nu in nu_list:
            print(f"\nTraining Sample {sample_idx+1}/{total_samples} | U_in: {u_in:.4f}, nu: {nu:.4f}")
            model = PINN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # 训练循环
            for epoch in range(3000):
                optimizer.zero_grad()
                
                # 【修复核心 1】将配点生成放入 Epoch 循环内部
                # 【修复核心 2】先进行数学运算(*10)，然后再赋予梯度属性，确保其是纯粹的叶子节点
                x_col = (torch.rand(2000, 1, device=device) * 10).requires_grad_(True)
                y_col = torch.rand(2000, 1, device=device).requires_grad_(True)

                # 边界点同样在循环内动态采样，防止模型对固定点过拟合
                x_wall = torch.rand(500, 1, device=device) * 10
                y_wall_bottom = torch.zeros(500, 1, device=device)
                y_wall_top = torch.ones(500, 1, device=device)
                
                y_inlet = torch.rand(500, 1, device=device)
                x_inlet = torch.zeros(500, 1, device=device)
                u_inlet_profile = 4 * u_in * y_inlet * (1 - y_inlet) # 抛物线分布

                y_outlet = torch.rand(500, 1, device=device)
                x_outlet = torch.ones(500, 1, device=device) * 10

                # 物理场 PDE Loss
                loss_pde = pde_loss(model, x_col, y_col, nu)
                
                # 边界条件 Loss
                out_wall_b = model(x_wall, y_wall_bottom)
                out_wall_t = model(x_wall, y_wall_top)
                loss_wall = torch.mean(out_wall_b[:, :2]**2) + torch.mean(out_wall_t[:, :2]**2)
                
                out_inlet = model(x_inlet, y_inlet)
                loss_inlet = torch.mean((out_inlet[:, 0:1] - u_inlet_profile)**2) + torch.mean(out_inlet[:, 1:2]**2)
                
                out_outlet = model(x_outlet, y_outlet)
                loss_outlet = torch.mean(out_outlet[:, 2:3]**2) # 压力 p=0
                
                loss = loss_pde + 10.0 * (loss_wall + loss_inlet + loss_outlet)
                loss.backward()
                optimizer.step()

            # 训练完成后，在网格上推理并保存数据
            model.eval()
            with torch.no_grad():
                pred = model(X_flat, Y_flat).cpu().numpy()
            
            u_pred = pred[:, 0].reshape(nx, ny)
            v_pred = pred[:, 1].reshape(nx, ny)
            p_pred = pred[:, 2].reshape(nx, ny)

            # 组装至整体数据集
            all_X[sample_idx] = X
            all_Y[sample_idx] = Y
            all_Q[sample_idx, 0] = u_pred
            all_Q[sample_idx, 1] = v_pred
            all_Q[sample_idx, 2] = p_pred
            all_Params[sample_idx] = [u_in, nu]

            sample_idx += 1

    # 4. 导出为 Transolver 支持的 .npy 格式
    os.makedirs("./channel_data", exist_ok=True)
    np.save("./channel_data/Channel_X.npy", all_X)
    np.save("./channel_data/Channel_Y.npy", all_Y)
    np.save("./channel_data/Channel_Q.npy", all_Q)
    np.save("./channel_data/Channel_Params.npy", all_Params)
    print("\n数据生成完毕并成功保存至 ./channel_data 目录。")

if __name__ == "__main__":
    generate_data()