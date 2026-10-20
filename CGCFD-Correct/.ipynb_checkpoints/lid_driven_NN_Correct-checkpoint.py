#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 顶盖驱动方腔流求解器 —— 基于神经网络代理模型的离散双残差误差补偿
====================================================================
核心流程:
  1. 密网格求解 (51×51) → 高精度参考解 (u_ref, v_ref, p_ref)
  2. 在粗网格节点 (21×21) 上采样密网格解并同步粗网格边界条件 → (u_s, v_s, p_s)
  3. 用粗网格自洽算子计算三项离散残差:
       Res_u   (u-动量截断残差)
       Res_v   (v-动量截断残差)
       Res_div (Rhie-Chow 连续性散度截断残差)
  4. 训练神经网络代理模型 NN: (x, y) -> (Res_u_NN, Res_v_NN, Res_div_NN)
     - 采用多尺度傅里叶特征 (Fourier Features) 克服顶盖角点奇异高频特征的频谱偏置
     - 采用 Z-score 标准化 + CosineAnnealingLR 余弦退火精确拟合
     - 预测时对 Res_div_NN 施加封闭域零均值约束 (sum = 0)，满足纯 Neumann 泊松方程可解性条件
  5. 将 NN 预测的源项注入粗网格求解器，对比 Original vs NN-Corrected 精度提升
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
# 五对角稀疏矩阵构建（向量化，coo_matrix 显式索引）
# ============================================================
def build_penta_matrix(aP, aE, aW, aN, aS):
    nix, niy = aP.shape
    n = nix * niy
    ii = np.arange(nix)[:, None] * niy + np.arange(niy)[None, :]
    rows_list, cols_list, vals_list = [], [], []

    # 主对角元
    rows_list.append(ii.ravel())
    cols_list.append(ii.ravel())
    vals_list.append(aP.ravel())

    # 东邻居 (i+1, j)
    rows_list.append(ii[:-1, :].ravel())
    cols_list.append((ii[:-1, :] + niy).ravel())
    vals_list.append(-aE[:-1, :].ravel())

    # 西邻居 (i-1, j)
    rows_list.append(ii[1:, :].ravel())
    cols_list.append((ii[1:, :] - niy).ravel())
    vals_list.append(-aW[1:, :].ravel())

    # 北邻居 (i, j+1)
    rows_list.append(ii[:, :-1].ravel())
    cols_list.append((ii[:, :-1] + 1).ravel())
    vals_list.append(-aN[:, :-1].ravel())

    # 南邻居 (i, j-1)
    rows_list.append(ii[:, 1:].ravel())
    cols_list.append((ii[:, 1:] - 1).ravel())
    vals_list.append(-aS[:, 1:].ravel())

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)

    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


# ============================================================
# 主求解器（支持动量+连续性双残差 NN 源项注入）
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

        self.rho = rho
        self.mu = mu
        self.u_lid = u_lid

        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.p = np.zeros((nx, ny))

        sh = (nx, ny)
        self.aE = np.zeros(sh); self.aW = np.zeros(sh)
        self.aN = np.zeros(sh); self.aS = np.zeros(sh)
        self.aP = np.zeros(sh); self.Sp = np.zeros(sh)

        self.sl = (slice(1, -1), slice(1, -1))

        # ===== 神经网络预测的离散残差源项 =====
        self.Sp_nn_u = None    # u 动量方程残差补偿
        self.Sp_nn_v = None    # v 动量方程残差补偿
        self.Sp_nn_div = None  # 连续性方程散度残差补偿

        self._init_fields()

    def _init_fields(self):
        self.u[:, :] = 0.0
        self.v[:, :] = 0.0
        self.p[:, :] = 0.0
        self._bc_u()
        self._bc_v()
        self._bc_p()

    def _bc_u(self):
        self.u[:, -1] = self.u_lid  # 顶盖运动
        self.u[0, :]  = 0.0         # 左壁
        self.u[-1, :] = 0.0         # 右壁
        self.u[:, 0]  = 0.0         # 下壁

    def _bc_v(self):
        self.v[0, :]  = 0.0
        self.v[-1, :] = 0.0
        self.v[:, 0]  = 0.0
        self.v[:, -1] = 0.0

    def _bc_p(self):
        self.p -= self.p[1, 1]      # 固定内部节点 (1,1) 为参考压力零点
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
        sl = self.sl

        ui = self.u[sl]
        vi = self.v[sl]

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
        sl = self.sl
        nx, ny = self.nx, self.ny
        niy = ny - 2
        nix = nx - 2

        phi_old = phi.copy()
        aP_ur = self.aP[sl] / alpha
        S_ur = self.Sp[sl] + (1.0 - alpha) / alpha * self.aP[sl] * phi_old[sl]

        rhs = S_ur.ravel().copy()

        rhs[0:niy]                 += self.aW[1, 1:-1]      * phi[0, 1:-1]
        rhs[(nix - 1) * niy:]      += self.aE[nx - 2, 1:-1] * phi[nx - 1, 1:-1]
        rhs[0:nix * niy:niy]       += self.aS[1:-1, 1]      * phi[1:-1, 0]
        rhs[niy - 1:nix * niy:niy] += self.aN[1:-1, ny - 2] * phi[1:-1, ny - 1]

        A = build_penta_matrix(aP_ur, self.aE[sl], self.aW[sl],
                               self.aN[sl], self.aS[sl])
        phi_internal = spsolve(A, rhs)
        phi[sl] = phi_internal.reshape((nix, niy), order='C')

    def _rc_face_velocities(self):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        d = np.zeros((nx, ny))
        d[1:-1, 1:-1] = 1.0 / self.aP[1:-1, 1:-1]
        d[0, :]  = d[1, :]
        d[-1, :] = d[-2, :]
        d[:, 0]  = d[:, 1]
        d[:, -1] = d[:, -2]

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
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        niy = ny - 2
        nix = nx - 2

        u_e, v_n, d = self._rc_face_velocities()

        div = np.zeros((nx, ny))
        sl = self.sl
        div[sl] = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / dx + \
                  (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / dy

        # 注入 NN 预测的连续性方程散度残差 Sp_nn_div
        if self.Sp_nn_div is not None:
            b = -(div[sl] - self.Sp_nn_div[sl]).copy()
        else:
            b = -div[sl].copy()

        aE_pc = 0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / dx ** 2
        aW_pc = 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / dx ** 2
        aN_pc = 0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / dy ** 2
        aS_pc = 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / dy ** 2
        aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

        # 壁面零梯度 Neumann BC: 主对角线减去越界系数
        aP_pc[0, :]  -= aW_pc[0, :]
        aP_pc[-1, :] -= aE_pc[-1, :]
        aP_pc[:, 0]  -= aS_pc[:, 0]
        aP_pc[:, -1] -= aN_pc[:, -1]

        # 固定内部节点 (0,0) 即全局 (1,1) 为参考点 p'=0
        aE_pc[0, 0] = 0.0
        aW_pc[0, 0] = 0.0
        aN_pc[0, 0] = 0.0
        aS_pc[0, 0] = 0.0
        b[0, 0] = 0.0

        A_p = build_penta_matrix(aP_pc, aE_pc, aW_pc, aN_pc, aS_pc)
        pp_internal = spsolve(A_p, b.ravel())

        pp = np.zeros((nx, ny))
        pp[1:-1, 1:-1] = pp_internal.reshape((nix, niy), order='C')
        self._bc_p_prime(pp)

        return pp

    def _correct_fields(self, pp, alpha_p):
        dx, dy = self.dx, self.dy
        sl = self.sl

        self.p[sl] += alpha_p * pp[sl]

        d = np.zeros((self.nx, self.ny))
        d[sl] = 1.0 / self.aP[sl]

        Gx_pp, Gy_pp = self._gradient(pp, dx, dy)
        self.u[sl] -= d[sl] * Gx_pp[sl]
        self.v[sl] -= d[sl] * Gy_pp[sl]

        self._bc_u()
        self._bc_v()
        self._bc_p()

    def _rel_change(self, phi_new, phi_old):
        denom = np.max(np.abs(phi_old))
        if denom < 1e-30:
            return np.mean(np.abs(phi_new - phi_old))
        return np.mean(np.abs(phi_new - phi_old)) / denom

    def solve(self, max_iter=3000, tol=1e-7,
              alpha_u=0.7, alpha_v=0.7, alpha_p=0.3, verbose=True):
        Re = self.rho * self.u_lid * self.L / self.mu
        if verbose:
            print("=" * 65)
            print(f"  2D 顶盖驱动方腔流 SIMPLE 求解器 | 网格: {self.nx} x {self.ny} | Re = {Re:.1f}")
            if self.Sp_nn_u is not None:
                print(f"  ✓ 启用神经网络代理模型双残差源项补偿 (Res_u + Res_v + Res_div)")
            print("=" * 65)

        u_old, v_old = self.u.copy(), self.v.copy()

        for it in range(1, max_iter + 1):
            self._assemble_momentum('u')
            self._solve_scalar(self.u, alpha_u)
            self._bc_u()

            self._assemble_momentum('v')
            self._solve_scalar(self.v, alpha_v)
            self._bc_v()

            pp = self._solve_pressure_correction()
            self._correct_fields(pp, alpha_p)

            res_u = self._rel_change(self.u, u_old)
            res_v = self._rel_change(self.v, v_old)
            res_p = np.mean(np.abs(pp[self.sl]))
            u_old, v_old = self.u.copy(), self.v.copy()

            if verbose and (it % 100 == 0 or it == 1):
                u_max = np.max(np.abs(self.u))
                psi_max = np.max(np.abs(self._streamfunc()))
                print(f"  iter {it:5d} | res_u = {res_u:.2e} | "
                      f"res_v = {res_v:.2e} | res_p = {res_p:.2e} | "
                      f"u_max = {u_max:.4f} | psi_max = {psi_max:.4f}")

            if res_u < tol and res_v < tol and res_p < tol and it > 10:
                if verbose:
                    print(f"  >> 收敛! iter = {it}, "
                          f"res_u = {res_u:.2e}, res_v = {res_v:.2e}, res_p = {res_p:.2e}")
                break
        else:
            if verbose:
                print(f"  >> 达到最大迭代次数 {max_iter}, "
                      f"res_u = {res_u:.2e}, res_v = {res_v:.2e}, res_p = {res_p:.2e}")

        return res_u, res_v, res_p

    def _streamfunc(self):
        psi = np.zeros((self.nx, self.ny))
        psi[:, 1:] = np.cumsum(0.5 * (self.u[:, 1:] + self.u[:, :-1]) * self.dy, axis=1)
        return psi


# ============================================================
# 完备离散残差计算器
# ============================================================
def compute_discrete_residual(solver, u_field, v_field, p_field):
    """
    给定密网格采样流场 (u, v, p)，在粗网格上严格执行边界条件后，
    计算动量方程残差 (res_u, res_v) 与连续性散度残差 (res_div)。
    """
    solver.u = u_field.copy()
    solver.v = v_field.copy()
    solver.p = p_field.copy()
    solver._bc_u()
    solver._bc_v()
    solver._bc_p()

    sl = solver.sl

    # 1. 组装粗网格动量方程系数
    solver._assemble_momentum('u')
    Gx, Gy = solver._gradient(solver.p, solver.dx, solver.dy)
    Sp_u = -Gx[sl]
    Sp_v = -Gy[sl]

    # 2. 计算动量残差: Res = aP*φ_P - Σ(a_nb*φ_nb) - Sp
    res_u = (solver.aP[sl] * solver.u[sl]
             - solver.aE[sl] * solver.u[2:, 1:-1]
             - solver.aW[sl] * solver.u[:-2, 1:-1]
             - solver.aN[sl] * solver.u[1:-1, 2:]
             - solver.aS[sl] * solver.u[1:-1, :-2]
             - Sp_u)

    res_v = (solver.aP[sl] * solver.v[sl]
             - solver.aE[sl] * solver.v[2:, 1:-1]
             - solver.aW[sl] * solver.v[:-2, 1:-1]
             - solver.aN[sl] * solver.v[1:-1, 2:]
             - solver.aS[sl] * solver.v[1:-1, :-2]
             - Sp_v)

    # 3. 计算采样场在粗网格 Rhie-Chow 界面速度下的离散散度残差
    u_e, v_n, _ = solver._rc_face_velocities()
    res_div = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / solver.dx + \
              (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / solver.dy

    return res_u, res_v, res_div, solver.u.copy(), solver.v.copy(), solver.p.copy()


# ============================================================
# 傅里叶特征神经网络代理模型 (克服方腔角点高频奇异性)
# ============================================================
class FourierMLP(nn.Module):
    """
    带多尺度傅里叶特征编码的全连接网络:
    将 (x, y) 映射到高频正余弦基函数空间，使网络能同时精准拟合顶盖角点剧烈奇异峰与中心平缓区。
    """
    def __init__(self, in_dim=2, hidden_dim=96, num_freqs=6):
        super().__init__()
        # 构造多尺度频率 [1, 2, 3, ..., num_freqs] * pi
        freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32) * np.pi
        self.register_buffer('freqs', freqs)
        feat_dim = in_dim + in_dim * num_freqs * 2

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: (N, 2)
        proj = x.unsqueeze(-1) * self.freqs  # (N, 2, num_freqs)
        proj = proj.reshape(x.shape[0], -1)
        feats = torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(feats)


class NeuralCorrector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.stats = {}  # 保存 Z-score 标准化参数 (mean, std)

    def prepare_training_data(self, solver_coarse, res_u, res_v, res_div):
        """
        准备训练数据:
        - 输入: 粗网格内部节点归一化坐标 (x/L, y/L)
        - 输出: 标准化后的三项离散残差 (Res_u, Res_v, Res_div)
        """
        X_in = solver_coarse.X[1:-1, 1:-1].ravel() / solver_coarse.L
        Y_in = solver_coarse.Y[1:-1, 1:-1].ravel() / solver_coarse.L
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

    def train(self, X_train, Y_tensors, epochs=4000, lr=3e-3):
        print(f"\n{'='*65}")
        print(f"Step 4: 训练傅里叶特征神经网络代理模型预测离散残差")
        print(f"  训练样本数: {len(X_train)} | 设备: {self.device}")
        print(f"  网络结构: FourierMLP (num_freqs=6, hidden=96×3, SiLU)")
        print(f"{'='*65}")

        criterion = nn.MSELoss()
        optimizers = {}
        schedulers = {}

        for k in ['u', 'v', 'div']:
            self.models[k] = FourierMLP(in_dim=2, hidden_dim=96, num_freqs=6).to(self.device)
            optimizers[k] = torch.optim.Adam(self.models[k].parameters(), lr=lr, weight_decay=1e-7)
            schedulers[k] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[k], T_max=epochs, eta_min=1e-6
            )

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
                print(f"  Epoch {epoch:4d} | Norm_MSE_u={losses['u']:.3e} | "
                      f"Norm_MSE_v={losses['v']:.3e} | Norm_MSE_div={losses['div']:.3e} | LR={cur_lr:.2e}")

        # 打印物理量纲下的最终拟合 RMSE
        with torch.no_grad():
            for k in ['u', 'v', 'div']:
                mean, std = self.stats[k]
                phys_rmse = np.sqrt(criterion(self.models[k](X_train), Y_tensors[k]).item()) * std
                print(f"  [训练集物理量纲 RMSE] Res_{k:3s} = {phys_rmse:.4e}")
        print("✓ 神经网络训练完成")

    def predict(self, solver):
        nx, ny = solver.nx, solver.ny
        X_all = np.column_stack([
            solver.X.ravel() / solver.L,
            solver.Y.ravel() / solver.L
        ])
        X_tensor = torch.tensor(X_all, dtype=torch.float32, device=self.device)

        preds = {}
        with torch.no_grad():
            for k in ['u', 'v', 'div']:
                mean, std = self.stats[k]
                raw = self.models[k](X_tensor).cpu().numpy().reshape(nx, ny)
                field = raw * std + mean
                # 边界节点无残差源项，强制置零
                field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0
                preds[k] = field

        # [关键物理约束]: 方腔为全封闭域，纯 Neumann 压力修正方程要求内部散度源项总和严格为 0
        # 减去内部节点预测值的微小均值漂移，保证全局质量严格守恒
        sl = solver.sl
        preds['div'][sl] -= np.mean(preds['div'][sl])

        return preds['u'], preds['v'], preds['div']


# ============================================================
# 可视化函数
# ============================================================
def plot_nn_source_terms(res_u_true, res_v_true, res_div_true,
                         res_u_pred, res_v_pred, res_div_pred, solver):
    """对比真实离散残差 vs 神经网络预测残差 vs 预测误差"""
    nx, ny = solver.nx, solver.ny
    true_fields = [res_u_true, res_v_true, res_div_true]
    pred_fields = [res_u_pred, res_v_pred, res_div_pred]
    names = ['Res_u (u-Momentum)', 'Res_v (v-Momentum)', 'Res_div (Continuity)']

    fig, axes = plt.subplots(3, 3, figsize=(17, 14))
    fig.suptitle("Discrete Residual Source Terms: Ground Truth vs NN Surrogate Prediction",
                 fontsize=14, fontweight='bold')

    for row, (t_in, p_full, name) in enumerate(zip(true_fields, pred_fields, names)):
        t_full = np.zeros((nx, ny))
        t_full[1:-1, 1:-1] = t_in
        diff = p_full - t_full

        for col, (data, sub_title) in enumerate([
            (t_full, f'True {name}'),
            (p_full, f'NN Predicted {name}'),
            (diff,   f'NN Error (max={np.max(np.abs(diff)):.2e})')
        ]):
            ax = axes[row, col]
            vmax = max(abs(data.min()), abs(data.max()), 1e-12)
            cf = ax.contourf(solver.X.T, solver.Y.T, data.T, levels=30,
                             cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            plt.colorbar(cf, ax=ax)
            ax.set_title(sub_title, fontsize=11)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('./lid_cavity_nn_source_terms.png', dpi=150, bbox_inches='tight')
    print("NN 源项拟合对比图已保存: ./lid_cavity_nn_source_terms.png")
    plt.show()


def plot_flow_field(solver_dense, solver_coarse_orig, solver_coarse_nn):
    """绘制流场对比：速度幅值云图 + 流线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Lid-Driven Cavity Flow: Dense Reference vs Coarse Original vs Coarse NN-Corrected",
                 fontsize=14, fontweight='bold')

    solvers = [solver_dense, solver_coarse_orig, solver_coarse_nn]
    titles = [
        f'Dense Grid Ref ({solver_dense.nx}×{solver_dense.ny})',
        f'Coarse Original ({solver_coarse_orig.nx}×{solver_coarse_orig.ny})',
        f'Coarse NN-Corrected ({solver_coarse_nn.nx}×{solver_coarse_nn.ny})'
    ]

    for col, (sv, title) in enumerate(zip(solvers, titles)):
        ax = axes[0, col]
        vel_mag = np.sqrt(sv.u**2 + sv.v**2)
        cf = ax.contourf(sv.X.T, sv.Y.T, vel_mag.T, levels=30, cmap='viridis',
                         vmin=0.0, vmax=1.0)
        plt.colorbar(cf, ax=ax, label='|V| (m/s)')
        ax.set_title(f'{title}\nVelocity Magnitude')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')

        ax = axes[1, col]
        psi = sv._streamfunc()
        levels = np.linspace(psi.min(), psi.max(), 30)
        cf = ax.contourf(sv.X.T, sv.Y.T, psi.T, levels=levels, cmap='Spectral_r')
        plt.colorbar(cf, ax=ax, label='ψ')
        ax.set_title(f'{title}\nStreamlines (ψ_min={psi.min():.4f})')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('./lid_cavity_nn_flow_field.png', dpi=150, bbox_inches='tight')
    print("流场对比图已保存: ./lid_cavity_nn_flow_field.png")
    plt.show()


def plot_profile_and_error(solver_dense, solver_coarse_orig, solver_coarse_nn,
                           u_sampled, v_sampled):
    """绘制中心线速度剖面与误差统计"""
    mid_i_d = solver_dense.nx // 2
    mid_j_d = solver_dense.ny // 2
    mid_i_c = solver_coarse_orig.nx // 2
    mid_j_c = solver_coarse_orig.ny // 2

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig1.suptitle("Centerline Velocity Profiles: Original vs NN-Corrected",
                  fontsize=13, fontweight='bold')

    # 垂直中心线 x=L/2: u(0.5, y)
    ax = axes1[0]
    ax.plot(solver_dense.u[mid_i_d, :], solver_dense.y, 'k-', lw=2.5, label='Dense Grid (Reference)')
    ax.plot(solver_coarse_orig.u[mid_i_c, :], solver_coarse_orig.y, 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
    ax.plot(solver_coarse_nn.u[mid_i_c, :], solver_coarse_nn.y, 'b:s', ms=5, lw=2.2, label='Coarse (NN-Corrected)')
    ax.set_xlabel('u (m/s)'); ax.set_ylabel('y (m)')
    ax.set_title('Vertical Centerline (x = L/2): u vs y')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 水平中心线 y=L/2: v(x, 0.5)
    ax = axes1[1]
    ax.plot(solver_dense.x, solver_dense.v[:, mid_j_d], 'k-', lw=2.5, label='Dense Grid (Reference)')
    ax.plot(solver_coarse_orig.x, solver_coarse_orig.v[:, mid_j_c], 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
    ax.plot(solver_coarse_nn.x, solver_coarse_nn.v[:, mid_j_c], 'b:s', ms=5, lw=2.2, label='Coarse (NN-Corrected)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('v (m/s)')
    ax.set_title('Horizontal Centerline (y = L/2): v vs x')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./lid_cavity_nn_centerline_profiles.png', dpi=150, bbox_inches='tight')
    print("中心线速度剖面图已保存: ./lid_cavity_nn_centerline_profiles.png")
    plt.show()

    # 误差统计 (对比采样参考解)
    err_u_orig = np.max(np.abs(u_sampled - solver_coarse_orig.u))
    err_u_nn   = np.max(np.abs(u_sampled - solver_coarse_nn.u))
    err_v_orig = np.max(np.abs(v_sampled - solver_coarse_orig.v))
    err_v_nn   = np.max(np.abs(v_sampled - solver_coarse_nn.v))

    rmse_u_orig = np.sqrt(np.mean((u_sampled - solver_coarse_orig.u)**2))
    rmse_u_nn   = np.sqrt(np.mean((u_sampled - solver_coarse_nn.u)**2))
    rmse_v_orig = np.sqrt(np.mean((v_sampled - solver_coarse_orig.v)**2))
    rmse_v_nn   = np.sqrt(np.mean((v_sampled - solver_coarse_nn.v)**2))

    imp_u_max = (1.0 - err_u_nn / err_u_orig) * 100
    imp_v_max = (1.0 - err_v_nn / err_v_orig) * 100
    imp_u_l2  = (1.0 - rmse_u_nn / rmse_u_orig) * 100
    imp_v_l2  = (1.0 - rmse_v_nn / rmse_v_orig) * 100

    print(f"\n  [最大绝对误差 Max Error]:")
    print(f"    u: 原粗网格 = {err_u_orig:.4e} → NN修正后 = {err_u_nn:.4e} (降低 {imp_u_max:.2f}%)")
    print(f"    v: 原粗网格 = {err_v_orig:.4e} → NN修正后 = {err_v_nn:.4e} (降低 {imp_v_max:.2f}%)")
    print(f"  [均方根误差 RMSE]:")
    print(f"    u: 原粗网格 = {rmse_u_orig:.4e} → NN修正后 = {rmse_u_nn:.4e} (降低 {imp_u_l2:.2f}%)")
    print(f"    v: 原粗网格 = {rmse_v_orig:.4e} → NN修正后 = {rmse_v_nn:.4e} (降低 {imp_v_l2:.2f}%)")

    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    x_pos = np.arange(2)
    width = 0.35
    bars1 = ax2.bar(x_pos - width/2, [err_u_orig, err_v_orig], width, label='Coarse Original', color='#ff6b6b')
    bars2 = ax2.bar(x_pos + width/2, [err_u_nn, err_v_nn], width, label='Coarse NN-Corrected', color='#4ecdc4')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['u-velocity Max Error', 'v-velocity Max Error'], fontsize=11)
    ax2.set_ylabel('Max Absolute Error (m/s)', fontsize=11)
    ax2.set_title(f'NN Surrogate Error Reduction: u = {imp_u_max:.1f}%, v = {imp_v_max:.1f}%',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_yscale('log')
    ax2.set_ylim(top=max(err_u_orig, err_v_orig) * 5.0)
    for bar, val in zip(list(bars1) + list(bars2), [err_u_orig, err_v_orig, err_u_nn, err_v_nn]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15, f'{val:.2e}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('./lid_cavity_nn_error_stats.png', dpi=150, bbox_inches='tight')
    print("误差统计柱状图已保存: ./lid_cavity_nn_error_stats.png")
    plt.show()

    return err_u_orig, err_u_nn, err_v_orig, err_v_nn


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    # ===== 配置参数 =====
    NX_DENSE, NY_DENSE   = 81, 81   # 密网格
    NX_COARSE, NY_COARSE = 11, 11   # 粗网格

    L     = 1.0
    RHO   = 1.0
    MU    = 0.01                    # Re = 100
    U_LID = 1.0

    # ===== Step 1: 密网格求解（参考解）=====
    print("\n" + "="*65)
    print("Step 1: 密网格求解 (参考解)")
    print("="*65)
    solver_dense = LidDrivenCavitySIMPLE(NX_DENSE, NY_DENSE, L, RHO, MU, U_LID)
    solver_dense.solve(max_iter=2000, tol=1e-7, alpha_u=0.7, alpha_v=0.7, alpha_p=0.3)

    # ===== Step 2: 在粗网格节点上采样密网格解 =====
    print("\n" + "="*65)
    print("Step 2: 在粗网格节点上采样密网格解")
    print("="*65)
    interp_u = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.u)
    interp_v = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.v)
    interp_p = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.p)

    solver_template = LidDrivenCavitySIMPLE(NX_COARSE, NY_COARSE, L, RHO, MU, U_LID)
    u_raw = interp_u(solver_template.x, solver_template.y)
    v_raw = interp_v(solver_template.x, solver_template.y)
    p_raw = interp_p(solver_template.x, solver_template.y)

    # ===== Step 3: 计算粗网格上的完备离散残差 (Res_u, Res_v, Res_div) =====
    print("\n" + "="*65)
    print("Step 3: 计算粗网格上的完备离散残差")
    print("="*65)
    res_u, res_v, res_div, u_sampled, v_sampled, p_sampled = compute_discrete_residual(
        solver_template, u_raw, v_raw, p_raw
    )
    print(f"  Res_u   范围: [{res_u.min():.4e}, {res_u.max():.4e}], mean|x|={np.mean(np.abs(res_u)):.4e}")
    print(f"  Res_v   范围: [{res_v.min():.4e}, {res_v.max():.4e}], mean|x|={np.mean(np.abs(res_v)):.4e}")
    print(f"  Res_div 范围: [{res_div.min():.4e}, {res_div.max():.4e}], mean|x|={np.mean(np.abs(res_div)):.4e}")

    # ===== Step 4: 训练神经网络代理模型 =====
    corrector = NeuralCorrector()
    X_train, Y_tensors = corrector.prepare_training_data(solver_template, res_u, res_v, res_div)
    corrector.train(X_train, Y_tensors, epochs=4000, lr=3e-3)

    # 预测粗网格全场残差源项并可视化拟合精度
    res_u_pred, res_v_pred, res_div_pred = corrector.predict(solver_template)
    plot_nn_source_terms(res_u, res_v, res_div, res_u_pred, res_v_pred, res_div_pred, solver_template)

    # ===== Step 5: 粗网格基准求解（无修正）=====
    print("\n" + "="*65)
    print("Step 5: 粗网格基准求解 (无修正)")
    print("="*65)
    solver_coarse_orig = LidDrivenCavitySIMPLE(NX_COARSE, NY_COARSE, L, RHO, MU, U_LID)
    solver_coarse_orig.solve(max_iter=2000, tol=1e-7, alpha_u=0.7, alpha_v=0.7, alpha_p=0.3)

    # ===== Step 6: 粗网格修正求解（注入 NN 预测的双残差源项）=====
    print("\n" + "="*65)
    print("Step 6: 粗网格修正求解 (注入 NN 预测的 Res_u + Res_v + Res_div)")
    print("="*65)
    solver_coarse_nn = LidDrivenCavitySIMPLE(NX_COARSE, NY_COARSE, L, RHO, MU, U_LID)
    solver_coarse_nn.Sp_nn_u   = res_u_pred
    solver_coarse_nn.Sp_nn_v   = res_v_pred
    solver_coarse_nn.Sp_nn_div = res_div_pred
    solver_coarse_nn.solve(max_iter=2000, tol=1e-7, alpha_u=0.7, alpha_v=0.7, alpha_p=0.3)

    # ===== Step 7: 误差对比分析与可视化 =====
    print("\n" + "="*65)
    print("Step 7: 误差对比分析与可视化")
    print("="*65)
    plot_profile_and_error(
        solver_dense, solver_coarse_orig, solver_coarse_nn, u_sampled, v_sampled
    )
    plot_flow_field(solver_dense, solver_coarse_orig, solver_coarse_nn)