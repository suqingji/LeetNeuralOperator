import torch
import torch.nn as nn
import numpy as np
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 优化 1 & 2：归一化坐标与 SiLU 激活函数
#class PINN(nn.Module):
#    def __init__(self):
#        super().__init__()
#        # 加深加宽一点，并使用 SiLU
#        self.net = nn.Sequential(
#            nn.Linear(2, 128), nn.SiLU(),
#            nn.Linear(128, 128), nn.SiLU(),
#            nn.Linear(128, 128), nn.SiLU(),
#            nn.Linear(128, 128), nn.SiLU(),
#            nn.Linear(128, 3)
#        )
#        
#    def forward(self, x, y):
#        # 将 x 从 [0, 10] 缩放到 [0, 1]，消除长宽比带来的梯度畸变
#        x_norm = x / 10.0
#        return self.net(torch.cat([x_norm, y], dim=1))
#class PINN(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.net = nn.Sequential(
#            nn.Linear(2, 64), nn.Tanh(),
#            nn.Linear(64, 64), nn.Tanh(),
#            nn.Linear(64, 64), nn.Tanh(),
#            nn.Linear(64, 64), nn.Tanh(),
#            nn.Linear(64, 3)
#        )
#        
#    def forward(self, x, y):
##        return self.net(torch.cat([x, y], dim=1))
#        x_norm = x / 10.0
#        return self.net(torch.cat([x_norm, y], dim=1))
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义三个完全独立的子网络，分别预测 u, v, p
        self.net_u = self._build_branch()
        self.net_v = self._build_branch()
        self.net_p = self._build_branch()
        
    def _build_branch(self):
        return nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, y):
        # 依旧保留坐标归一化
        x_norm = x / 10.0
        xy = torch.cat([x_norm, y], dim=1)
        
        u = self.net_u(xy)
        v = self.net_v(xy)
        p = self.net_p(xy)
        
        return torch.cat([u, v, p], dim=1)

        
def pde_loss(model, x, y, nu):
    out = model(x, y)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]

    eq_continuity = u_x + v_y
    eq_momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    eq_momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return torch.mean(eq_continuity**2) + torch.mean(eq_momentum_x**2) + torch.mean(eq_momentum_y**2)

def generate_data(num_samples, epochs):
    nx, ny = 400, 40
    x_grid = np.linspace(0, 10, nx)
    y_grid = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    X_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device)
    Y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32, device=device)

    # 动态初始化数组维度
    all_X = np.zeros((num_samples, nx, ny))
    all_Y = np.zeros((num_samples, nx, ny))
    all_Q = np.zeros((num_samples, 3, nx, ny))
    all_Params = np.zeros((num_samples, 2))

    for sample_idx in range(num_samples):
        # 随机分配物理参数，或固定单个参数
        u_in = np.random.uniform(5.0, 15.0) if num_samples > 1 else 5.0
        nu = np.random.uniform(0.01, 0.1) if num_samples > 1 else 0.05
        
        print(f"\nTraining Sample {sample_idx+1}/{num_samples} | U_in: {u_in:.4f}, nu: {nu:.4f}")
        model = PINN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            x_col = (torch.rand(2000, 1, device=device) * 10).requires_grad_(True)
            y_col = torch.rand(2000, 1, device=device).requires_grad_(True)
            x_wall = torch.rand(500, 1, device=device) * 10
            y_wall_bottom = torch.zeros(500, 1, device=device)
            y_wall_top = torch.ones(500, 1, device=device)
            
            y_inlet = torch.rand(500, 1, device=device)
            x_inlet = torch.zeros(500, 1, device=device)
            #u_inlet_profile = 4 * u_in * y_inlet * (1 - y_inlet)
            u_inlet_profile = 4 * u_in * y_inlet * (1 - y_inlet)
            
            y_outlet = torch.rand(500, 1, device=device)
            x_outlet = torch.ones(500, 1, device=device) * 10

            loss_pde = pde_loss(model, x_col, y_col, nu)
            
            out_wall_b = model(x_wall, y_wall_bottom)
            out_wall_t = model(x_wall, y_wall_top)
            loss_wall = torch.mean(out_wall_b[:, :2]**2) + torch.mean(out_wall_t[:, :2]**2)
            
            out_inlet = model(x_inlet, y_inlet)
            loss_inlet = torch.mean((out_inlet[:, 0:1] - u_inlet_profile)**2) + torch.mean(out_inlet[:, 1:2]**2)
            
            out_outlet = model(x_outlet, y_outlet)
            loss_outlet = torch.mean(out_outlet[:, 2:3]**2)
            
            loss = loss_pde + 10.0 * (loss_wall + loss_inlet + loss_outlet)
            loss.backward()
            optimizer.step()

            # ====== 在此处补充以下代码 ======
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:4d}/{epochs} | Total Loss: {loss.item():.6f} | PDE Loss: {loss_pde.item():.6f}")
            # =================================

        model.eval()
        with torch.no_grad():
            pred = model(X_flat, Y_flat).cpu().numpy()
        
        all_X[sample_idx] = X
        all_Y[sample_idx] = Y
        all_Q[sample_idx, 0] = pred[:, 0].reshape(nx, ny)
        all_Q[sample_idx, 1] = pred[:, 1].reshape(nx, ny)
        all_Q[sample_idx, 2] = pred[:, 2].reshape(nx, ny)
        all_Params[sample_idx] = [u_in, nu]

    os.makedirs("./channel_data", exist_ok=True)
    np.save("./channel_data/Channel_X.npy", all_X)
    np.save("./channel_data/Channel_Y.npy", all_Y)
    np.save("./channel_data/Channel_Q.npy", all_Q)
    np.save("./channel_data/Channel_Params.npy", all_Params)
    print("\n数据生成完毕并成功保存至 ./channel_data 目录。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3000)
    args = parser.parse_args()
    generate_data(args.num_samples, args.epochs)