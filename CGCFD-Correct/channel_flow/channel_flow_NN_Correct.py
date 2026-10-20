#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 槽道流求解器 —— 基于离散残差的神经网络误差补偿 (完整修复版)
================================================================
修复要点:
  1. 修复底层 SIMPLE 求解器的 3 个 Bug (TDMA 边界移项、压力修正边界系数与内迭代).
  2. 引入【动量 + 连续性(散度)】双残差补偿机制 (Dual-Consistency):
     不仅补偿粗网格动量方程截断误差 (Res_u, Res_v),
     同时补偿粗网格 Rhie-Chow 连续性方程截断误差 (Res_div), 防止压力修正冲毁动量修正.
  3. 保持采样场压力边界与粗网格算子自洽.
  4. 为神经网络增加目标值 Z-score 标准化与余弦退火学习率衰减 (CosineAnnealingLR).
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import RectBivariateSpline


# ============================================================
# 向量化 Thomas 算法
# ============================================================
def tdma_batch(low, diag, up, rhs):
    N, M = rhs.shape
    cp = np.zeros_like(rhs)
    dp = np.zeros_like(rhs)
    cp[:, 0] = up[:, 0] / diag[:, 0]
    dp[:, 0] = rhs[:, 0] / diag[:, 0]
    for k in range(1, M):
        w = diag[:, k] - low[:, k] * cp[:, k - 1]
        cp[:, k] = up[:, k] / w
        dp[:, k] = (rhs[:, k] - low[:, k] * dp[:, k - 1]) / w
    x = np.zeros_like(rhs)
    x[:, -1] = dp[:, -1]
    for k in range(M - 2, -1, -1):
        x[:, k] = dp[:, k] - cp[:, k] * x[:, k + 1]
    return x


# ============================================================
# 主求解器 (修复完备版)
# ============================================================
class ChannelFlowSIMPLE:
    def __init__(self, nx=51, ny=21, Lx=10.0, Ly=1.0,
                 rho=1.0, mu=0.01, u_in=1.0):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.rho = rho
        self.mu = mu
        self.u_in = u_in

        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.p = np.zeros((nx, ny))

        sh = (nx, ny)
        self.aE = np.zeros(sh); self.aW = np.zeros(sh)
        self.aN = np.zeros(sh); self.aS = np.zeros(sh)
        self.aP = np.zeros(sh); self.Sp = np.zeros(sh)

        # 神经网络源项（动量残差 + 连续性散度残差补偿）
        self.Sp_nn_u = None
        self.Sp_nn_v = None
        self.Sp_nn_div = None

        self._init_fields()

    def _init_fields(self):
        self.u[:, :] = self.u_in
        self.v[:, :] = 0.0
        self.p[:, :] = 0.0
        self.u[:, 0] = self.u[:, -1] = 0.0
        self.u[0, 0] = self.u[-1, 0] = self.u[0, -1] = self.u[-1, -1] = 0.0

    def _bc_u(self):
        self.u[0, 1:-1] = self.u_in
        self.u[-1, 1:-1] = self.u[-2, 1:-1]
        self.u[:, 0] = self.u[:, -1] = 0.0

    def _bc_v(self):
        self.v[0, :] = 0.0
        self.v[-1, :] = self.v[-2, :]
        self.v[:, 0] = self.v[:, -1] = 0.0

    def _bc_p(self):
        self.p[-1, :] = 0.0
        self.p[0, :] = self.p[1, :]
        self.p[:, 0] = self.p[:, 1]
        self.p[:, -1] = self.p[:, -2]

    def _bc_p_prime(self, pp):
        pp[-1, :] = 0.0
        pp[0, :] = pp[1, :]
        pp[:, 0] = pp[:, 1]
        pp[:, -1] = pp[:, -2]

    @staticmethod
    def _gradient(f, dx, dy):
        Gx = np.zeros_like(f)
        Gy = np.zeros_like(f)
        Gx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
        Gy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
        Gx[0, :]  = (-3 * f[0, :]  + 4 * f[1, :]  - f[2, :])  / (2 * dx)
        Gx[-1, :] = ( 3 * f[-1, :] - 4 * f[-2, :] + f[-3, :]) / (2 * dx)
        Gy[:, 0]  = (-3 * f[:, 0]  + 4 * f[:, 1]  - f[:, 2])  / (2 * dy)
        Gy[:, -1] = ( 3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * dy)
        return Gx, Gy

    def _assemble_momentum(self, direction):
        dx, dy = self.dx, self.dy
        rho, mu = self.rho, self.mu
        sl = (slice(1, -1), slice(1, -1))
        ui = self.u[sl]; vi = self.v[sl]

        De = Dw = mu / dx ** 2
        Dn = Ds = mu / dy ** 2

        aE_c = np.where(ui >= 0, 0.0, -rho * ui / dx)
        aW_c = np.where(ui >= 0,  rho * ui / dx, 0.0)
        aN_c = np.where(vi >= 0, 0.0, -rho * vi / dy)
        aS_c = np.where(vi >= 0,  rho * vi / dy, 0.0)

        self.aE[sl] = De + aE_c
        self.aW[sl] = Dw + aW_c
        self.aN[sl] = Dn + aN_c
        self.aS[sl] = Ds + aS_c
        self.aP[sl] = self.aE[sl] + self.aW[sl] + self.aN[sl] + self.aS[sl]

        Gx, Gy = self._gradient(self.p, dx, dy)
        if direction == 'u':
            self.Sp[sl] = -Gx[sl]
            if self.Sp_nn_u is not None:
                self.Sp[sl] += self.Sp_nn_u[sl]
        else:
            self.Sp[sl] = -Gy[sl]
            if self.Sp_nn_v is not None:
                self.Sp[sl] += self.Sp_nn_v[sl]

    def _solve_scalar(self, phi, alpha):
        sl = (slice(1, -1), slice(1, -1))
        phi_old = phi.copy()
        aP_ur = self.aP.copy()
        S_ur  = self.Sp.copy()
        aP_ur[sl] = self.aP[sl] / alpha
        S_ur[sl]  = self.Sp[sl] + (1.0 - alpha) / alpha * self.aP[sl] * phi_old[sl]

        # --- x 扫掠 (包含边界移项修复) ---
        low  = -self.aW[sl].T.copy()
        diag = aP_ur[sl].T.copy()
        up   = -self.aE[sl].T.copy()
        rhs  = (self.aN[sl] * phi[1:-1, 2:] +
                self.aS[sl] * phi[1:-1, :-2] +
                S_ur[sl]).T
        rhs[:, 0]   += self.aW[1, 1:-1] * phi[0, 1:-1]
        diag[:, -1] -= self.aE[-2, 1:-1]

        phi[1:-1, 1:-1] = tdma_batch(low, diag, up, rhs).T
        phi[-1, 1:-1] = phi[-2, 1:-1]

        # --- y 扫掠 (包含边界移项修复) ---
        low  = -self.aS[sl].copy()
        diag = aP_ur[sl].copy()
        up   = -self.aN[sl].copy()
        rhs  = (self.aE[sl] * phi[2:, 1:-1] +
                self.aW[sl] * phi[:-2, 1:-1] +
                S_ur[sl])
        rhs[:, 0]  += self.aS[1:-1, 1] * phi[1:-1, 0]
        rhs[:, -1] += self.aN[1:-1, -2] * phi[1:-1, -1]

        phi[1:-1, 1:-1] = tdma_batch(low, diag, up, rhs)

    def _rc_face_velocities(self, alpha_u):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        d = np.zeros((nx, ny))
        d[1:-1, 1:-1] = alpha_u / self.aP[1:-1, 1:-1]
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

    def _solve_pressure_correction(self, alpha_u, n_pc_sweeps=25):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        u_e, v_n, d = self._rc_face_velocities(alpha_u)

        div = np.zeros((nx, ny))
        sl = (slice(1, nx - 1), slice(1, ny - 1))
        div[sl] = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / dx + \
                  (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / dy

        # [核心修复]: 注入连续性方程的离散散度残差 Sp_nn_div
        if self.Sp_nn_div is not None:
            b = -(div[sl] - self.Sp_nn_div[sl])
        else:
            b = -div[sl]

        aE_pc = 0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / dx ** 2
        aW_pc = 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / dx ** 2
        aN_pc = 0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / dy ** 2
        aS_pc = 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / dy ** 2
        aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

        # [修复]: 正确处理零梯度边界并清零越界系数
        aP_pc[0, :]  -= aW_pc[0, :];  aW_pc[0, :]  = 0.0
        aP_pc[:, 0]  -= aS_pc[:, 0];  aS_pc[:, 0]  = 0.0
        aP_pc[:, -1] -= aN_pc[:, -1]; aN_pc[:, -1] = 0.0
        aE_pc[-1, :] = 0.0

        pp = np.zeros((nx, ny))
        low_x, diag_x, up_x = -aW_pc.T, aP_pc.T, -aE_pc.T
        low_y, diag_y, up_y = -aS_pc,   aP_pc,   -aN_pc

        for _ in range(n_pc_sweeps):
            rhs_x = (aN_pc * pp[1:-1, 2:] + aS_pc * pp[1:-1, :-2] + b).T
            pp[1:-1, 1:-1] = tdma_batch(low_x, diag_x, up_x, rhs_x).T
            rhs_y = (aE_pc * pp[2:, 1:-1] + aW_pc * pp[:-2, 1:-1] + b)
            pp[1:-1, 1:-1] = tdma_batch(low_y, diag_y, up_y, rhs_y)

        self._bc_p_prime(pp)
        mass_resid = np.max(np.abs(b))
        return pp, d, mass_resid

    def _correct_fields(self, pp, d, alpha_p):
        dx, dy = self.dx, self.dy
        sl = (slice(1, -1), slice(1, -1))
        self.p[sl] += alpha_p * pp[sl]
        Gx_pp, Gy_pp = self._gradient(pp, dx, dy)
        self.u[sl] -= d[sl] * Gx_pp[sl]
        self.v[sl] -= d[sl] * Gy_pp[sl]
        self._bc_u(); self._bc_v(); self._bc_p()

    def solve(self, max_iter=4000, tol=1e-6,
              alpha_u=0.6, alpha_v=0.6, alpha_p=0.2, verbose=True):
        if verbose:
            print("=" * 60)
            print(f"  2D 槽道流 SIMPLE 求解器 | 网格: {self.nx} x {self.ny}")
            if self.Sp_nn_u is not None:
                print(f"  ✓ 启用神经网络双残差源项补偿 (Momentum + Continuity)")
            print("=" * 60)

        for it in range(1, max_iter + 1):
            self._assemble_momentum('u')
            self._solve_scalar(self.u, alpha_u)
            self._bc_u()

            self._assemble_momentum('v')
            self._solve_scalar(self.v, alpha_v)
            self._bc_v()

            pp, d, mass_resid = self._solve_pressure_correction(alpha_u)
            self._correct_fields(pp, d, alpha_p)

            if verbose and (it % 200 == 0 or it == 1):
                u_max = np.max(np.abs(self.u))
                print(f"  iter {it:5d} | max(div_err) = {mass_resid:.2e} | "
                      f"u_max = {u_max:.4f} | p_range = [{np.min(self.p):.2f}, {np.max(self.p):.2f}]")

            if mass_resid < tol and it > 15:
                if verbose:
                    print(f"  >> 收敛! iter = {it}, max(div_err) = {mass_resid:.2e}")
                break
        else:
            if verbose:
                print(f"  >> 达到最大迭代次数 {max_iter}, max(div_err) = {mass_resid:.2e}")
        return mass_resid


# ============================================================
# 完备离散残差计算器 (动量 + 连续性)
# ============================================================
def compute_discrete_residual(solver, u_s, v_s, p_s, alpha_u=0.6):
    """
    给定密网格采样流场 (u_s, v_s, p_s)，用粗网格求解器完全自洽的算子计算离散残差。
    返回内部节点的:
      Res_u:   u-动量离散残差
      Res_v:   v-动量离散残差
      Res_div: Rhie-Chow 连续性方程离散散度残差
    """
    # 1. 将采样场载入粗网格模板并同步粗网格边界条件 (确保算子严格一致!)
    solver.u = u_s.copy()
    solver.v = v_s.copy()
    solver.p = p_s.copy()
    solver._bc_u()
    solver._bc_v()
    solver._bc_p()

    sl = (slice(1, -1), slice(1, -1))

    # 2. 组装粗网格动量系数 (此时 solver.Sp_nn_u 为 None)
    solver._assemble_momentum('u')
    Gx, Gy = solver._gradient(solver.p, solver.dx, solver.dy)
    Sp_u = -Gx[sl]
    Sp_v = -Gy[sl]

    # 保持出口零梯度与 _solve_scalar 一致
    u_eval = solver.u.copy()
    v_eval = solver.v.copy()
    u_eval[-1, 1:-1] = u_eval[-2, 1:-1]
    v_eval[-1, 1:-1] = v_eval[-2, 1:-1]

    # 计算动量残差: Res = aP*phi_P - sum(a_nb*phi_nb) - Sp
    Res_u = (solver.aP[sl] * u_eval[sl]
             - solver.aE[sl] * u_eval[2:, 1:-1]
             - solver.aW[sl] * u_eval[:-2, 1:-1]
             - solver.aN[sl] * u_eval[1:-1, 2:]
             - solver.aS[sl] * u_eval[1:-1, :-2]
             - Sp_u)

    Res_v = (solver.aP[sl] * v_eval[sl]
             - solver.aE[sl] * v_eval[2:, 1:-1]
             - solver.aW[sl] * v_eval[:-2, 1:-1]
             - solver.aN[sl] * v_eval[1:-1, 2:]
             - solver.aS[sl] * v_eval[1:-1, :-2]
             - Sp_v)

    # 3. 计算采样场在粗网格 Rhie-Chow 格式下的离散散度残差 Res_div
    u_e, v_n, _ = solver._rc_face_velocities(alpha_u)
    Res_div = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / solver.dx + \
              (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / solver.dy

    return Res_u, Res_v, Res_div


# ============================================================
# 神经网络误差补偿器 (带标准化与余弦退火)
# ============================================================
class NeuralCorrector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.stats = {}  # 保存每个目标的均值和标准差 (mean, std)

    def build_model(self, input_dim=2, hidden_layers=[64, 64]):
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers).to(self.device)

    def prepare_training_data(self, solver_coarse, res_u, res_v, res_div):
        # 向量化提取内部节点归一化坐标
        X_in = solver_coarse.X[1:-1, 1:-1].ravel() / solver_coarse.Lx
        Y_in = solver_coarse.Y[1:-1, 1:-1].ravel() / solver_coarse.Ly
        X_train = torch.tensor(np.column_stack([X_in, Y_in]), dtype=torch.float32, device=self.device)

        targets = {'u': res_u.ravel(), 'v': res_v.ravel(), 'div': res_div.ravel()}
        Y_tensors = {}
        for k, val in targets.items():
            mean = float(np.mean(val))
            std = float(np.std(val)) + 1e-12
            self.stats[k] = (mean, std)
            norm_val = (val - mean) / std
            Y_tensors[k] = torch.tensor(norm_val, dtype=torch.float32, device=self.device).unsqueeze(1)

        return X_train, Y_tensors

    def train(self, X_train, Y_tensors, epochs=3000, lr=2e-3):
        print(f"\n{'='*60}")
        print(f"训练神经网络预测离散残差 (样本数: {len(X_train)})")
        print(f"{'='*60}")

        criterion = nn.MSELoss()
        optimizers = {}
        schedulers = {}

        for k in ['u', 'v', 'div']:
            self.models[k] = self.build_model(hidden_layers=[64, 64])
            optimizers[k] = torch.optim.Adam(self.models[k].parameters(), lr=lr, weight_decay=1e-6)
            schedulers[k] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[k], T_max=epochs, eta_min=1e-5)

        # 全批量训练 (粗网格内部节点通常仅几百个，全批量比 mini-batch 快 10 倍且梯度更稳)
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
                print(f"  Epoch {epoch:4d} | MSE_u={losses['u']:.3e} | "
                      f"MSE_v={losses['v']:.3e} | MSE_div={losses['div']:.3e} | LR={cur_lr:.2e}")

        print("✓ 神经网络训练完成\n")

    def predict(self, solver):
        nx, ny = solver.nx, solver.ny
        X_all = np.column_stack([
            solver.X.ravel() / solver.Lx,
            solver.Y.ravel() / solver.Ly
        ])
        X_tensor = torch.tensor(X_all, dtype=torch.float32, device=self.device)

        preds = {}
        with torch.no_grad():
            for k in ['u', 'v', 'div']:
                mean, std = self.stats[k]
                raw = self.models[k](X_tensor).cpu().numpy().reshape(nx, ny)
                field = raw * std + mean
                # 边界节点残差置零
                field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0
                preds[k] = field

        return preds['u'], preds['v'], preds['div']


# ============================================================
# 可视化函数
# ============================================================
def plot_all_comparisons(solver_dense, solver_coarse_orig, solver_coarse_corrected,
                         res_u_true, res_u_pred):
    interp_u = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.u)
    u_ref_on_coarse = interp_u(solver_coarse_orig.x, solver_coarse_orig.y)

    # 1. 源项 Res_u 对比
    nx, ny = solver_coarse_orig.nx, solver_coarse_orig.ny
    res_u_full = np.zeros((nx, ny))
    res_u_full[1:-1, 1:-1] = res_u_true

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 4.5))
    fig1.suptitle("Discrete Residual Source Term (Res_u): True vs NN Predicted", fontsize=14, fontweight='bold')

    vmin, vmax = res_u_full.min(), res_u_full.max()
    for ax, data, title in zip(
        axes1,
        [res_u_full, res_u_pred, res_u_pred - res_u_full],
        ['True Res_u', 'NN Predicted Res_u', f'Error (max={np.max(np.abs(res_u_pred - res_u_full)):.2e})']
    ):
        cf = ax.contourf(solver_coarse_orig.X.T, solver_coarse_orig.Y.T, data.T, levels=25, cmap='RdBu_r')
        plt.colorbar(cf, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('./source_term_res_u_analysis.png', dpi=150, bbox_inches='tight')

    # 2. 流场与剖面综合对比
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle("2D Channel Flow: Coarse Grid NN Defect Correction Results", fontsize=14, fontweight='bold')

    # (a) 密网格云图
    ax = axes2[0, 0]
    cf = ax.contourf(solver_dense.X.T, solver_dense.Y.T, solver_dense.u.T, levels=30, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='u (m/s)')
    ax.set_title(f'Dense Grid Reference ({solver_dense.nx}×{solver_dense.ny})')
    ax.set_aspect('equal')

    # (b) 粗网格修正后云图
    ax = axes2[0, 1]
    cf = ax.contourf(solver_coarse_corrected.X.T, solver_coarse_corrected.Y.T,
                     solver_coarse_corrected.u.T, levels=30, cmap='viridis',
                     vmin=solver_dense.u.min(), vmax=solver_dense.u.max())
    plt.colorbar(cf, ax=ax, label='u (m/s)')
    ax.set_title(f'Coarse Grid NN-Corrected ({solver_coarse_corrected.nx}×{solver_coarse_corrected.ny})')
    ax.set_aspect('equal')

    # (c) x = 7.5m 截面速度剖面对比
    ax = axes2[1, 0]
    i_dense = np.argmin(np.abs(solver_dense.x - 7.5))
    i_coarse = np.argmin(np.abs(solver_coarse_orig.x - 7.5))
    ax.plot(solver_dense.u[i_dense, :], solver_dense.y, 'k-', lw=2.5, label='Dense Grid (Reference)')
    ax.plot(solver_coarse_orig.u[i_coarse, :], solver_coarse_orig.y, 'r--o', ms=5, lw=1.8, label='Coarse Original')
    ax.plot(solver_coarse_corrected.u[i_coarse, :], solver_coarse_corrected.y, 'b:s', ms=5, lw=2.2, label='Coarse NN-Corrected')
    ax.set_xlabel('u (m/s)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Velocity Profile at x = {solver_coarse_orig.x[i_coarse]:.2f} m')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (d) 截面绝对误差分布 (对数坐标)
    ax = axes2[1, 1]
    err_orig = np.abs(u_ref_on_coarse - solver_coarse_orig.u)
    err_corr = np.abs(u_ref_on_coarse - solver_coarse_corrected.u)
    ax.semilogy(solver_coarse_orig.y, err_orig[i_coarse, :] + 1e-12, 'r-o', lw=2, label='Original Coarse Error')
    ax.semilogy(solver_coarse_orig.y, err_corr[i_coarse, :] + 1e-12, 'b-s', lw=2, label='NN-Corrected Error')
    ax.set_xlabel('y (m)'); ax.set_ylabel('|u - u_ref| (m/s)')
    ax.set_title('Pointwise Absolute Error at x = 7.5 m')
    ax.legend(); ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('./flow_field_and_profile_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    # 建议使用奇数网格节点，确保粗/密网格嵌套且包含中心线 y = 0.5
    NX_DENSE, NY_DENSE   = 81, 41
    NX_COARSE, NY_COARSE = 11, 6

    LX, LY = 10.0, 1.0
    RHO, MU, U_IN = 1.0, 0.01, 1.0
    ALPHA_U, ALPHA_V, ALPHA_P = 0.6, 0.6, 0.2

    # ===== Step 1: 密网格求解 =====
    print("\n" + "="*60)
    print("Step 1: 密网格求解 (参考解)")
    print("="*60)
    solver_dense = ChannelFlowSIMPLE(NX_DENSE, NY_DENSE, LX, LY, RHO, MU, U_IN)
    solver_dense.solve(max_iter=4000, tol=1e-6, alpha_u=ALPHA_U, alpha_v=ALPHA_V, alpha_p=ALPHA_P)

    # ===== Step 2: 在粗网格上采样密网格解 =====
    print("\n" + "="*60)
    print("Step 2: 在粗网格节点上采样密网格解")
    print("="*60)
    interp_u = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.u)
    interp_v = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.v)
    interp_p = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.p)

    solver_coarse_template = ChannelFlowSIMPLE(NX_COARSE, NY_COARSE, LX, LY, RHO, MU, U_IN)
    u_sampled = interp_u(solver_coarse_template.x, solver_coarse_template.y)
    v_sampled = interp_v(solver_coarse_template.x, solver_coarse_template.y)
    p_sampled = interp_p(solver_coarse_template.x, solver_coarse_template.y)

    # ===== Step 3: 计算完备离散残差 (动量 + 连续性) =====
    print("\n" + "="*60)
    print("Step 3: 计算粗网格上的离散残差 (Res_u, Res_v, Res_div)")
    print("="*60)
    res_u, res_v, res_div = compute_discrete_residual(
        solver_coarse_template, u_sampled, v_sampled, p_sampled, alpha_u=ALPHA_U
    )
    print(f"  Res_u   范围: [{res_u.min():.4e}, {res_u.max():.4e}]")
    print(f"  Res_v   范围: [{res_v.min():.4e}, {res_v.max():.4e}]")
    print(f"  Res_div 范围: [{res_div.min():.4e}, {res_div.max():.4e}]")

    # ===== Step 4: 训练神经网络 =====
    corrector = NeuralCorrector()
    X_train, Y_tensors = corrector.prepare_training_data(solver_coarse_template, res_u, res_v, res_div)
    corrector.train(X_train, Y_tensors, epochs=3000, lr=2e-3)

    # ===== Step 5: 粗网格基准求解（无修正）=====
    print("\n" + "="*60)
    print("Step 5: 粗网格基准求解 (无修正)")
    print("="*60)
    solver_coarse_orig = ChannelFlowSIMPLE(NX_COARSE, NY_COARSE, LX, LY, RHO, MU, U_IN)
    solver_coarse_orig.solve(max_iter=4000, tol=1e-6, alpha_u=ALPHA_U, alpha_v=ALPHA_V, alpha_p=ALPHA_P)

    # ===== Step 6: 粗网格修正求解 =====
    print("\n" + "="*60)
    print("Step 6: 粗网格修正求解 (注入 NN 预测的双残差源项)")
    print("="*60)
    solver_coarse_corrected = ChannelFlowSIMPLE(NX_COARSE, NY_COARSE, LX, LY, RHO, MU, U_IN)
    res_u_pred, res_v_pred, res_div_pred = corrector.predict(solver_coarse_corrected)
    solver_coarse_corrected.Sp_nn_u = res_u_pred
    solver_coarse_corrected.Sp_nn_v = res_v_pred
    solver_coarse_corrected.Sp_nn_div = res_div_pred
    solver_coarse_corrected.solve(max_iter=4000, tol=1e-6, alpha_u=ALPHA_U, alpha_v=ALPHA_V, alpha_p=ALPHA_P)

    # ===== Step 7: 误差分析 (仅评估内部流场节点) =====
    print("\n" + "="*60)
    print("Step 7: 误差对比分析")
    print("="*60)
    err_orig_max = np.max(np.abs(u_sampled[1:-1, 1:-1] - solver_coarse_orig.u[1:-1, 1:-1]))
    err_corr_max = np.max(np.abs(u_sampled[1:-1, 1:-1] - solver_coarse_corrected.u[1:-1, 1:-1]))
    err_orig_l2  = np.sqrt(np.mean((u_sampled[1:-1, 1:-1] - solver_coarse_orig.u[1:-1, 1:-1])**2))
    err_corr_l2  = np.sqrt(np.mean((u_sampled[1:-1, 1:-1] - solver_coarse_corrected.u[1:-1, 1:-1])**2))

    imp_max = (1.0 - err_corr_max / err_orig_max) * 100
    imp_l2  = (1.0 - err_corr_l2  / err_orig_l2)  * 100

    print(f"  [Max Error] 原始粗网格: {err_orig_max:.6f} -> NN修正后: {err_corr_max:.6f} (降低 {imp_max:.2f}%)")
    print(f"  [RMSE     ] 原始粗网格: {err_orig_l2:.6f} -> NN修正后: {err_corr_l2:.6f} (降低 {imp_l2:.2f}%)")

    # ===== 可视化 =====
    plot_all_comparisons(solver_dense, solver_coarse_orig, solver_coarse_corrected, res_u, res_u_pred)