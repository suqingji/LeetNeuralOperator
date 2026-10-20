#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 顶盖驱动方腔流求解器 —— 泛化傅里叶神经网络多网格测试版
====================================================================
改进要点:
  1. 修复底层矩阵和算法的语法格式错误。
  2. 扩展 FourierMLP 输入维度至 4 维 (x, y, dx, dy)，支持跨网格粗糙度学习。
  3. 支持对多个未见过的测试网格进行泛化评估。
  4. 绘制方腔流流场与叠加物理网格线的对比图，绘制标准中心线速度剖面。
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RectBivariateSpline

# 固定随机种子以保证可复现性
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 五对角稀疏矩阵构建
# ============================================================
def build_penta_matrix(aP, aE, aW, aN, aS):
    nix, niy = aP.shape
    n = nix * niy
    ii = np.arange(nix)[:, None] * niy + np.arange(niy)[None, :]
    rows_list, cols_list, vals_list = [], [], []

    rows_list.append(ii.ravel())
    cols_list.append(ii.ravel())
    vals_list.append(aP.ravel())

    rows_list.append(ii[:-1, :].ravel())
    cols_list.append((ii[:-1, :] + niy).ravel())
    vals_list.append(-aE[:-1, :].ravel())

    rows_list.append(ii[1:, :].ravel())
    cols_list.append((ii[1:, :] - niy).ravel())
    vals_list.append(-aW[1:, :].ravel())

    rows_list.append(ii[:, :-1].ravel())
    cols_list.append((ii[:, :-1] + 1).ravel())
    vals_list.append(-aN[:, :-1].ravel())

    rows_list.append(ii[:, 1:].ravel())
    cols_list.append((ii[:, 1:] - 1).ravel())
    vals_list.append(-aS[:, 1:].ravel())

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


# ============================================================
# 主求解器（修复语法截断问题）
# ============================================================
class LidDrivenCavitySIMPLE:
    def __init__(self, nx=51, ny=51, L=1.0, rho=1.0, mu=0.01, u_lid=1.0):
        self.nx, self.ny = nx, ny
        self.L = L
        self.dx = L / (nx - 1)
        self.dy = L / (ny - 1)
        self.x = np.linspace(0, L, nx)
        self.y = np.linspace(0, L, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        self.rho, self.mu, self.u_lid = rho, mu, u_lid
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.p = np.zeros((nx, ny))

        sh = (nx, ny)
        self.aE = np.zeros(sh); self.aW = np.zeros(sh)
        self.aN = np.zeros(sh); self.aS = np.zeros(sh)
        self.aP = np.zeros(sh); self.Sp = np.zeros(sh)
        self.sl = (slice(1, -1), slice(1, -1))

        self.Sp_nn_u = None   
        self.Sp_nn_v = None   
        self.Sp_nn_div = None 
        self._init_fields()

    def _init_fields(self):
        self.u[:, :] = 0.0; self.v[:, :] = 0.0; self.p[:, :] = 0.0
        self._bc_u(); self._bc_v(); self._bc_p()

    def _bc_u(self):
        self.u[:, -1] = self.u_lid  
        self.u[0, :]  = 0.0         
        self.u[-1, :] = 0.0         
        self.u[:, 0]  = 0.0         

    def _bc_v(self):
        self.v[0, :]  = 0.0; self.v[-1, :] = 0.0
        self.v[:, 0]  = 0.0; self.v[:, -1] = 0.0

    def _bc_p(self):
        self.p -= self.p[1, 1]      
        self.p[0, :]  = self.p[1, :]
        self.p[-1, :] = self.p[-2, :]
        self.p[:, 0]  = self.p[:, 1]
        self.p[:, -1] = self.p[:, -2]

    def _bc_p_prime(self, pp):
        pp -= pp[1, 1]
        pp[0, :]  = pp[1, :]
        pp[-1, :] = pp[-2, :]
        pp[:, 0]  = pp[:, 1]
        pp[:, -1] = pp[:, -2]

    @staticmethod
    def _gradient(f, dx, dy):
        Gx = np.zeros_like(f); Gy = np.zeros_like(f)
        Gx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
        Gy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
        Gx[0, :]  = (-3 * f[0, :]  + 4 * f[1, :]  - f[2, :])  / (2 * dx)
        Gx[-1, :] = ( 3 * f[-1, :] - 4 * f[-2, :] + f[-3, :]) / (2 * dx)
        Gy[:, 0]  = (-3 * f[:, 0]  + 4 * f[:, 1]  - f[:, 2])  / (2 * dy)
        Gy[:, -1] = ( 3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * dy)
        return Gx, Gy

    def _assemble_momentum(self, direction):
        dx, dy, rho, mu = self.dx, self.dy, self.rho, self.mu
        sl = self.sl; ui = self.u[sl]; vi = self.v[sl]

        De = Dw = mu / dx ** 2
        Dn = Ds = mu / dy ** 2

        aE_c = np.where(ui >= 0, 0.0, -rho * ui / dx)
        aW_c = np.where(ui >= 0,  rho * ui / dx, 0.0)
        aN_c = np.where(vi >= 0, 0.0, -rho * vi / dy)
        aS_c = np.where(vi >= 0,  rho * vi / dy, 0.0)

        self.aE[sl] = De + aE_c; self.aW[sl] = Dw + aW_c
        self.aN[sl] = Dn + aN_c; self.aS[sl] = Ds + aS_c
        self.aP[sl] = self.aE[sl] + self.aW[sl] + self.aN[sl] + self.aS[sl]

        Gx, Gy = self._gradient(self.p, dx, dy)
        if direction == 'u':
            self.Sp[sl] = -Gx[sl]
            if self.Sp_nn_u is not None: self.Sp[sl] += self.Sp_nn_u[sl]
        else:
            self.Sp[sl] = -Gy[sl]
            if self.Sp_nn_v is not None: self.Sp[sl] += self.Sp_nn_v[sl]

    def _solve_scalar(self, phi, alpha):
        sl = self.sl; nx, ny = self.nx, self.ny
        niy, nix = ny - 2, nx - 2

        phi_old = phi.copy()
        aP_ur = self.aP[sl] / alpha
        S_ur = self.Sp[sl] + (1.0 - alpha) / alpha * self.aP[sl] * phi_old[sl]

        rhs = S_ur.ravel().copy()
        rhs[0:niy]                 += self.aW[1, 1:-1]      * phi[0, 1:-1]
        rhs[(nix - 1) * niy:]      += self.aE[nx - 2, 1:-1] * phi[nx - 1, 1:-1]
        rhs[0:nix * niy:niy]       += self.aS[1:-1, 1]      * phi[1:-1, 0]
        rhs[niy - 1:nix * niy:niy] += self.aN[1:-1, ny - 2] * phi[1:-1, ny - 1]

        A = build_penta_matrix(aP_ur, self.aE[sl], self.aW[sl], self.aN[sl], self.aS[sl])
        phi_internal = spsolve(A, rhs)
        phi[sl] = phi_internal.reshape((nix, niy), order='C')

    def _rc_face_velocities(self):
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        d = np.zeros((nx, ny))
        d[1:-1, 1:-1] = 1.0 / self.aP[1:-1, 1:-1]
        d[0, :] = d[1, :]; d[-1, :] = d[-2, :]
        d[:, 0] = d[:, 1]; d[:, -1] = d[:, -2]

        Gx, Gy = self._gradient(self.p, dx, dy)

        d_bar_e = 0.5 * (d[:-1, :] + d[1:, :])
        Gx_bar_e = 0.5 * (Gx[:-1, :] + Gx[1:, :])
        Gx_compact_e = (self.p[1:, :] - self.p[:-1, :]) / dx
        u_e = 0.5 * (self.u[:-1, :] + self.u[1:, :]) + d_bar_e * (Gx_bar_e - Gx_compact_e)

        d_bar_n = 0.5 * (d[:, :-1] + d[:, 1:])
        Gy_bar_n = 0.5 * (Gy[:, :-1] + Gy[:, 1:])
        Gy_compact_n = (self.p[:, 1:] - self.p[:, :-1]) / dy
        v_n = 0.5 * (self.v[:, :-1] + self.v[:, 1:]) + d_bar_n * (Gy_bar_n - Gy_compact_n)

        return u_e, v_n, d

    def _solve_pressure_correction(self):
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        niy, nix = ny - 2, nx - 2
        u_e, v_n, d = self._rc_face_velocities()

        div = np.zeros((nx, ny)); sl = self.sl
        div[sl] = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / dx + (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / dy

        b = -(div[sl] - self.Sp_nn_div[sl]).copy() if self.Sp_nn_div is not None else -div[sl].copy()

        aE_pc = 0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / dx ** 2
        aW_pc = 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / dx ** 2
        aN_pc = 0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / dy ** 2
        aS_pc = 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / dy ** 2
        aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

        aP_pc[0, :]  -= aW_pc[0, :]; aP_pc[-1, :] -= aE_pc[-1, :]
        aP_pc[:, 0]  -= aS_pc[:, 0]; aP_pc[:, -1] -= aN_pc[:, -1]

        aE_pc[0, 0] = 0.0; aW_pc[0, 0] = 0.0
        aN_pc[0, 0] = 0.0; aS_pc[0, 0] = 0.0; b[0, 0] = 0.0

        A_p = build_penta_matrix(aP_pc, aE_pc, aW_pc, aN_pc, aS_pc)
        pp_internal = spsolve(A_p, b.ravel())

        pp = np.zeros((nx, ny))
        pp[1:-1, 1:-1] = pp_internal.reshape((nix, niy), order='C')
        self._bc_p_prime(pp)
        return pp

    def _correct_fields(self, pp, alpha_p):
        dx, dy, sl = self.dx, self.dy, self.sl
        self.p[sl] += alpha_p * pp[sl]
        d = np.zeros((self.nx, self.ny)); d[sl] = 1.0 / self.aP[sl]
        Gx_pp, Gy_pp = self._gradient(pp, dx, dy)
        self.u[sl] -= d[sl] * Gx_pp[sl]; self.v[sl] -= d[sl] * Gy_pp[sl]
        self._bc_u(); self._bc_v(); self._bc_p()

    def _rel_change(self, phi_new, phi_old):
        denom = np.max(np.abs(phi_old))
        if denom < 1e-30: return np.mean(np.abs(phi_new - phi_old))
        return np.mean(np.abs(phi_new - phi_old)) / denom

    def solve(self, max_iter=3000, tol=1e-7, alpha_u=0.7, alpha_v=0.7, alpha_p=0.3, verbose=True):
        u_old, v_old = self.u.copy(), self.v.copy()
        if verbose: print(f"[{self.nx}x{self.ny}] 方腔流求解中...", end=" ")
        
        for it in range(1, max_iter + 1):
            self._assemble_momentum('u'); self._solve_scalar(self.u, alpha_u); self._bc_u()
            self._assemble_momentum('v'); self._solve_scalar(self.v, alpha_v); self._bc_v()
            pp = self._solve_pressure_correction()
            self._correct_fields(pp, alpha_p)

            res_u = self._rel_change(self.u, u_old); res_v = self._rel_change(self.v, v_old)
            res_p = np.mean(np.abs(pp[self.sl]))
            u_old, v_old = self.u.copy(), self.v.copy()

            if res_u < tol and res_v < tol and res_p < tol and it > 10:
                if verbose: print(f"收敛于 {it} 步 (res_u={res_u:.2e})")
                break
        else:
            if verbose: print(f"达到最大迭代次数 (res_u={res_u:.2e})")
        return res_u, res_v, res_p


# ============================================================
# 完备离散残差计算器
# ============================================================
def compute_discrete_residual(solver, u_field, v_field, p_field):
    solver.u, solver.v, solver.p = u_field.copy(), v_field.copy(), p_field.copy()
    solver._bc_u(); solver._bc_v(); solver._bc_p()
    sl = solver.sl

    solver._assemble_momentum('u')
    Gx, Gy = solver._gradient(solver.p, solver.dx, solver.dy)
    Sp_u, Sp_v = -Gx[sl], -Gy[sl]

    res_u = (solver.aP[sl] * solver.u[sl] - solver.aE[sl] * solver.u[2:, 1:-1] - solver.aW[sl] * solver.u[:-2, 1:-1]
             - solver.aN[sl] * solver.u[1:-1, 2:] - solver.aS[sl] * solver.u[1:-1, :-2] - Sp_u)
    res_v = (solver.aP[sl] * solver.v[sl] - solver.aE[sl] * solver.v[2:, 1:-1] - solver.aW[sl] * solver.v[:-2, 1:-1]
             - solver.aN[sl] * solver.v[1:-1, 2:] - solver.aS[sl] * solver.v[1:-1, :-2] - Sp_v)

    u_e, v_n, _ = solver._rc_face_velocities()
    res_div = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / solver.dx + (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / solver.dy

    return res_u, res_v, res_div


# ============================================================
# 泛化多尺度傅里叶特征代理模型
# ============================================================
class GeneralizedFourierMLP(nn.Module):
    """
    输入特征: (x, y, dx, dy)。通过傅里叶编码捕捉空间高频奇异性。
    """
    def __init__(self, in_dim=4, hidden_dim=96, num_freqs=6):
        super().__init__()
        freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32) * np.pi
        self.register_buffer('freqs', freqs)
        feat_dim = in_dim + in_dim * num_freqs * 2

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        proj = x.unsqueeze(-1) * self.freqs  
        proj = proj.reshape(x.shape[0], -1)
        feats = torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(feats)


class GeneralizedNeuralCorrector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.stats = {} 

    def prepare_multi_grid_data(self, datasets):
        X_all = np.vstack([d['X'] for d in datasets])
        X_train = torch.tensor(X_all, dtype=torch.float32, device=self.device)

        targets = {
            'u': np.concatenate([d['u'] for d in datasets]),
            'v': np.concatenate([d['v'] for d in datasets]),
            'div': np.concatenate([d['div'] for d in datasets])
        }

        Y_tensors = {}
        for k, val in targets.items():
            mean, std = float(np.mean(val)), float(np.std(val)) + 1e-12
            self.stats[k] = (mean, std)
            norm_val = (val - mean) / std
            Y_tensors[k] = torch.tensor(norm_val, dtype=torch.float32, device=self.device).unsqueeze(1)
            
        return X_train, Y_tensors

    def train(self, X_train, Y_tensors, epochs=4000, lr=3e-3):
        print(f"\n[{'='*60}]")
        print(f"开始训练泛化多尺度傅里叶特征代理模型 (样本数: {len(X_train)})")
        print(f"[{'='*60}]")

        criterion = nn.MSELoss()
        optimizers = {}; schedulers = {}

        for k in ['u', 'v', 'div']:
            self.models[k] = GeneralizedFourierMLP(in_dim=4, hidden_dim=96, num_freqs=6).to(self.device)
            optimizers[k] = torch.optim.Adam(self.models[k].parameters(), lr=lr, weight_decay=1e-7)
            schedulers[k] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[k], T_max=epochs, eta_min=1e-6)

        for epoch in range(1, epochs + 1):
            losses = {}
            for k in ['u', 'v', 'div']:
                optimizers[k].zero_grad()
                pred = self.models[k](X_train)
                loss = criterion(pred, Y_tensors[k])
                loss.backward()
                optimizers[k].step()
                schedulers[k].step()
                losses[k] = loss.item()

            if epoch % 500 == 0 or epoch == 1:
                cur_lr = optimizers['u'].param_groups[0]['lr']
                print(f"  Epoch {epoch:4d} | MSE_u={losses['u']:.3e} | MSE_v={losses['v']:.3e} | "
                      f"MSE_div={losses['div']:.3e} | LR={cur_lr:.2e}")

    def predict(self, solver):
        nx, ny = solver.nx, solver.ny
        X_norm, Y_norm = solver.X.ravel() / solver.L, solver.Y.ravel() / solver.L
        dx_norm, dy_norm = solver.dx / solver.L, solver.dy / solver.L

        X_all = np.column_stack([
            X_norm, Y_norm,
            np.full_like(X_norm, dx_norm),
            np.full_like(X_norm, dy_norm)
        ])
        X_tensor = torch.tensor(X_all, dtype=torch.float32, device=self.device)

        preds = {}
        with torch.no_grad():
            for k in ['u', 'v', 'div']:
                mean, std = self.stats[k]
                raw = self.models[k](X_tensor).cpu().numpy().reshape(nx, ny)
                field = raw * std + mean
                field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0
                preds[k] = field

        # [封闭域约束]: 保障全局守恒性，泊松可解
        preds['div'][solver.sl] -= np.mean(preds['div'][solver.sl])
        return preds['u'], preds['v'], preds['div']


# ============================================================
# 测试组泛化独立绘图函数
# ============================================================
def plot_generalization_results(solver_dense, solver_test_orig, solver_test_corr, test_grid):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Lid-Driven Cavity: Zero-Shot Generalization Test on {test_grid}", 
                 fontsize=15, fontweight='bold')
    
    # 获取中心线索引 (确保网络形状是奇数，便于取严格的中心剖面)
    mid_i_d, mid_j_d = solver_dense.nx // 2, solver_dense.ny // 2
    mid_i_c, mid_j_c = solver_test_orig.nx // 2, solver_test_orig.ny // 2

    # --- 1. 原始粗网格云图 (带网格线) ---
    ax = axes[0, 0]
    vel_mag1 = np.sqrt(solver_test_orig.u**2 + solver_test_orig.v**2)
    cf1 = ax.contourf(solver_test_orig.X.T, solver_test_orig.Y.T, vel_mag1.T, levels=30, cmap='viridis')
    plt.colorbar(cf1, ax=ax, label='|V| (m/s)')
    # 叠加物理网格线
    for x_val in solver_test_orig.x: ax.axvline(x_val, color='w', lw=0.5, alpha=0.4)
    for y_val in solver_test_orig.y: ax.axhline(y_val, color='w', lw=0.5, alpha=0.4)
    ax.set_title(f'Original Coarse ({test_grid[0]}x{test_grid[1]}) + Physical Mesh')
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

    # --- 2. 神经网络修正云图 (带网格线) ---
    ax = axes[0, 1]
    vel_mag2 = np.sqrt(solver_test_corr.u**2 + solver_test_corr.v**2)
    cf2 = ax.contourf(solver_test_corr.X.T, solver_test_corr.Y.T, vel_mag2.T, levels=30, cmap='viridis')
    plt.colorbar(cf2, ax=ax, label='|V| (m/s)')
    # 叠加物理网格线
    for x_val in solver_test_corr.x: ax.axvline(x_val, color='w', lw=0.5, alpha=0.4)
    for y_val in solver_test_corr.y: ax.axhline(y_val, color='w', lw=0.5, alpha=0.4)
    ax.set_title(f'NN-Corrected Coarse ({test_grid[0]}x{test_grid[1]}) + Physical Mesh')
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

    # --- 3. 垂直中心线 x=L/2: u 速度剖面 ---
    ax = axes[1, 0]
    ax.plot(solver_dense.u[mid_i_d, :], solver_dense.y, 'k-', lw=2.5, label='Dense Reference')
    ax.plot(solver_test_orig.u[mid_i_c, :], solver_test_orig.y, 'r--o', ms=5, label='Original Coarse')
    ax.plot(solver_test_corr.u[mid_i_c, :], solver_test_corr.y, 'b:s', ms=5, label='NN-Corrected')
    ax.set_title("Vertical Centerline (x=L/2): u-velocity")
    ax.set_xlabel('u (m/s)'); ax.set_ylabel('y (m)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- 4. 水平中心线 y=L/2: v 速度剖面 ---
    ax = axes[1, 1]
    ax.plot(solver_dense.x, solver_dense.v[:, mid_j_d], 'k-', lw=2.5, label='Dense Reference')
    ax.plot(solver_test_orig.x, solver_test_orig.v[:, mid_j_c], 'r--o', ms=5, label='Original Coarse')
    ax.plot(solver_test_corr.x, solver_test_corr.v[:, mid_j_c], 'b:s', ms=5, label='NN-Corrected')
    ax.set_title("Horizontal Centerline (y=L/2): v-velocity")
    ax.set_xlabel('x (m)'); ax.set_ylabel('v (m/s)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'./Cavity_Generalization_{test_grid[0]}x{test_grid[1]}.png', dpi=150)
    plt.show()


# ============================================================
# 主执行流
# ============================================================
if __name__ == '__main__':
    L, RHO, MU, U_LID = 1.0, 1.0, 0.01, 1.0  # Re = 100
    ALPHA_U, ALPHA_V, ALPHA_P = 0.7, 0.7, 0.3

    # 密网格基准
    DENSE_GRID = (81, 81)
    
    # 提供给神经网络学习的多尺度特征网格
    TRAIN_GRIDS = [(11, 11), (15, 15), (21, 21), (31, 31)]
    
    # 需要验证的未知粗糙度网格测试集 (使用奇数网格便于严格提取中心剖面)
    TEST_GRIDS = [
        (13, 13), # 极粗 (位于 11x11 和 15x15 之间)
        (17, 17), # 中粗 (位于 15x15 和 21x21 之间)
        (25, 25)  # 较密 (位于 21x21 和 31x31 之间)
    ]

    print(f"\n>>> Step 1: 密网格基准求解 {DENSE_GRID} <<<")
    solver_dense = LidDrivenCavitySIMPLE(*DENSE_GRID, L, RHO, MU, U_LID)
    solver_dense.solve(max_iter=2000)
    interp_u = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.u)
    interp_v = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.v)
    interp_p = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.p)

    print(f"\n>>> Step 2: 收集多尺度粗网格特征与真实离散残差 <<<")
    datasets = []
    for (nx, ny) in TRAIN_GRIDS:
        print(f" -> 处理训练集网格: {nx} x {ny}")
        solver_c = LidDrivenCavitySIMPLE(nx, ny, L, RHO, MU, U_LID)
        u_s = interp_u(solver_c.x, solver_c.y)
        v_s = interp_v(solver_c.x, solver_c.y)
        p_s = interp_p(solver_c.x, solver_c.y)
        res_u, res_v, res_div = compute_discrete_residual(solver_c, u_s, v_s, p_s)
        
        # 提取当前网格的 4 维特征 (x, y, dx, dy)
        X_norm = solver_c.X[1:-1, 1:-1].ravel() / L
        Y_norm = solver_c.Y[1:-1, 1:-1].ravel() / L
        dx_norm, dy_norm = solver_c.dx / L, solver_c.dy / L
        
        X_features = np.column_stack([
            X_norm, Y_norm, 
            np.full_like(X_norm, dx_norm), np.full_like(Y_norm, dy_norm)
        ])
        
        datasets.append({'X': X_features, 'u': res_u.ravel(), 'v': res_v.ravel(), 'div': res_div.ravel()})

    print(f"\n>>> Step 3: 训练通用多尺度代理模型 <<<")
    corrector = GeneralizedNeuralCorrector()
    X_train, Y_tensors = corrector.prepare_multi_grid_data(datasets)
    corrector.train(X_train, Y_tensors, epochs=4000, lr=3e-3)

    print("\n" + "="*65)
    print(">>> Step 4: 循环测试未见过的网格并进行可视化分析 <<<")
    print("="*65)

    for test_grid in TEST_GRIDS:
        print(f"\n>>> [启动泛化测试] 网格: {test_grid} <<<")
        
        # 1. 原始粗网格求解
        solver_test_orig = LidDrivenCavitySIMPLE(*test_grid, L, RHO, MU, U_LID)
        solver_test_orig.solve(verbose=False)
        
        # 2. 神经网络修正粗网格求解
        solver_test_corr = LidDrivenCavitySIMPLE(*test_grid, L, RHO, MU, U_LID)
        res_u_pred, res_v_pred, res_div_pred = corrector.predict(solver_test_corr)
        solver_test_corr.Sp_nn_u = res_u_pred
        solver_test_corr.Sp_nn_v = res_v_pred
        solver_test_corr.Sp_nn_div = res_div_pred
        solver_test_corr.solve(verbose=False)

        # 3. 计算误差并打印评估结果
        u_ref = interp_u(solver_test_orig.x, solver_test_orig.y)
        v_ref = interp_v(solver_test_orig.x, solver_test_orig.y)
        
        err_u_orig = np.max(np.abs(u_ref[1:-1, 1:-1] - solver_test_orig.u[1:-1, 1:-1]))
        err_u_corr = np.max(np.abs(u_ref[1:-1, 1:-1] - solver_test_corr.u[1:-1, 1:-1]))
        err_v_orig = np.max(np.abs(v_ref[1:-1, 1:-1] - solver_test_orig.v[1:-1, 1:-1]))
        err_v_corr = np.max(np.abs(v_ref[1:-1, 1:-1] - solver_test_corr.v[1:-1, 1:-1]))
        
        print(f"  [评估] u 速度最大误差 - 原粗网格: {err_u_orig:.4f}  |  NN修正: {err_u_corr:.4f} (降低 {(1-err_u_corr/err_u_orig)*100:.1f}%)")
        print(f"  [评估] v 速度最大误差 - 原粗网格: {err_v_orig:.4f}  |  NN修正: {err_v_corr:.4f} (降低 {(1-err_v_corr/err_v_orig)*100:.1f}%)")

        # 4. 独立调用绘图生成图像并保存
        plot_generalization_results(solver_dense, solver_test_orig, solver_test_corr, test_grid)
        print("-" * 65)