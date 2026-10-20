import torch
import torch.nn as nn
import numpy as np
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    def __init__(self, x_scale=1.0):
        super().__init__()
        self.x_scale = x_scale
        # 推荐使用共享特征层的多层全连接网络，有助于捕捉流场变量间的强耦合
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 3)
        )
        
    def forward(self, x, y):
        # 动态缩放 x，消除区域长宽比带来的畸变
        x_norm = x / self.x_scale
        out = self.net(torch.cat([x_norm, y], dim=1))
        return out

        
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
    # Lid-driven cavity configuration
    nx, ny = 64, 64  # grid resolution suitable for cavity
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    X_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device)
    Y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32, device=device)
    
    # Allocate storage for multiple samples
    all_X = np.zeros((num_samples, nx, ny))
    all_Y = np.zeros((num_samples, nx, ny))
    all_Q = np.zeros((num_samples, 3, nx, ny))
    all_Params = np.zeros((num_samples, 2))  # store U_lid and ν
    
    for sample_idx in range(num_samples):
        # Randomize lid speed and viscosity for each sample (or fix for single sample)
        U_lid = np.random.uniform(0.5, 2.0) if num_samples > 1 else 1.0
        nu = np.random.uniform(0.01, 0.1) if num_samples > 1 else 0.005
        
        print(f"\nTraining Sample {sample_idx+1}/{num_samples} | U_lid: {U_lid:.4f}, nu: {nu:.4f}")
        # 实例化时传入对应的 x_scale
        model = PINN(x_scale=1.0).to(device)  
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # 引入指数学习率衰减
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
        
        # 建议 epochs 增加至至少 10000
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Collocation points for PDE residual
            x_col = torch.rand(2000, 1, device=device)
            y_col = torch.rand(2000, 1, device=device)
            x_col.requires_grad_(True)
            y_col.requires_grad_(True)
            
            # Boundary points: bottom (y=0), left (x=0), right (x=1) – no-slip (u=v=0)
            xb = torch.rand(300, 1, device=device)
            yb = torch.zeros(300, 1, device=device)
            xl = torch.zeros(300, 1, device=device)
            yl = torch.rand(300, 1, device=device)
            xr = torch.ones(300, 1, device=device)
            yr = torch.rand(300, 1, device=device)
            
            # Top lid (y=1) – u=U_lid, v=0
            xt = torch.rand(300, 1, device=device)
            yt = torch.ones(300, 1, device=device)
            
            loss_pde = pde_loss(model, x_col, y_col, nu)
            
            out_bottom = model(xb, yb)
            out_left = model(xl, yl)
            out_right = model(xr, yr)
            out_top = model(xt, yt)
            
            # No-slip on bottom, left, right (both u and v zero)
            loss_ns = (
                torch.mean(out_bottom[:, :2] ** 2) +
                torch.mean(out_left[:, :2] ** 2) +
                torch.mean(out_right[:, :2] ** 2)
            )
            # Lid velocity on top: u=U_lid, v=0
            loss_lid = torch.mean((out_top[:, 0:1] - U_lid) ** 2) + torch.mean(out_top[:, 1:2] ** 2)

            # ====== 新增：压力锚点 ======
            # 强制原点 (0,0) 的压力 p=0，消除系统自由度
            p_ref = model(torch.zeros(1, 1, device=device), torch.zeros(1, 1, device=device))[:, 2:3]
            loss_p_ref = torch.mean(p_ref ** 2)
            # ============================
            
            loss = loss_pde + 100.0 * (loss_ns + loss_lid) + loss_p_ref
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if epoch % 200 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:4d}/{epochs} | Total Loss: {loss.item():.6f} | PDE Loss: {loss_pde.item():.6f}")
        
        # Save predictions on the full grid
        model.eval()
        with torch.no_grad():
            pred = model(X_flat, Y_flat).cpu().numpy()
        
        all_X[sample_idx] = X
        all_Y[sample_idx] = Y
        all_Q[sample_idx, 0] = pred[:, 0].reshape(nx, ny)  # u
        all_Q[sample_idx, 1] = pred[:, 1].reshape(nx, ny)  # v
        all_Q[sample_idx, 2] = pred[:, 2].reshape(nx, ny)  # p
        all_Params[sample_idx] = [U_lid, nu]
    
    os.makedirs("./cavity_data", exist_ok=True)
    np.save("./cavity_data/Cavity_X.npy", all_X)
    np.save("./cavity_data/Cavity_Y.npy", all_Y)
    np.save("./cavity_data/Cavity_Q.npy", all_Q)
    np.save("./cavity_data/Cavity_Params.npy", all_Params)
    print("\n数据生成完毕并成功保存至 ./cavity_data 目录。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10000)
    args = parser.parse_args()
    generate_data(args.num_samples, args.epochs)