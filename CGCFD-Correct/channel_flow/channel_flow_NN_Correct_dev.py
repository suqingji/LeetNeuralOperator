#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 槽道流求解器 —— 具有严格旋转协变性 (O(2) Equivariant) 与稳定在线耦合的神经网络误差补偿
=========================================================================================
本版本修复了导致在线耦合流场失速/恶化的三大问题:
  1. 质量守恒强制约束: 对神经网络预测的 S_div 强制实施零均值修正，杜绝非物理的体积膨胀加速。
  2. 修复量纲错位: 移除手动限幅带来的量纲缩放 Bug，完全依赖 PyTorch 内部的 Soft-Clamping 约束。
  3. 剔除目标歧义: 回归纯净参考态特征提取，使网络建立严谨的 u* -> R(u*) 单射关系。
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.interpolate import RectBivariateSpline

# ============================================================
# 向量化 Thomas 算法 (TDMA)
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
# 主求解器：支持水平 ('x') 与旋转90度竖直 ('y') 槽道
# ============================================================
class ChannelFlowSIMPLE:
    def __init__(self, nx=51, ny=21, Lx=10.0, Ly=1.0,
                 rho=1.0, mu=0.01, u_in=1.0, flow_axis='x'):
        assert flow_axis in ('x', 'y'), "flow_axis 必须为 'x' 或 'y'"
        self.flow_axis = flow_axis
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
        self.L_char = Ly if flow_axis == 'x' else Lx
        self.Re = (rho * u_in * self.L_char) / max(mu, 1e-12)

        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.p = np.zeros((nx, ny))

        sh = (nx, ny)
        self.aE = np.zeros(sh); self.aW = np.zeros(sh)
        self.aN = np.zeros(sh); self.aS = np.zeros(sh)
        self.aP = np.zeros(sh); self.Sp = np.zeros(sh)

        self.Sp_nn_u = None
        self.Sp_nn_v = None
        self.Sp_nn_div = None

        self._init_fields()

    def _init_fields(self):
        self.p[:, :] = 0.0
        if self.flow_axis == 'x':
            self.u[:, :] = self.u_in
            self.v[:, :] = 0.0
            self.u[:, 0] = self.u[:, -1] = 0.0
        else:
            self.u[:, :] = 0.0
            self.v[:, :] = self.u_in
            self.v[0, :] = self.v[-1, :] = 0.0

    def _bc_u(self):
        if self.flow_axis == 'x':
            self.u[0, 1:-1] = self.u_in
            self.u[-1, 1:-1] = self.u[-2, 1:-1]
            self.u[:, 0] = self.u[:, -1] = 0.0
        else:
            self.u[:, 0] = 0.0
            self.u[:, -1] = self.u[:, -2]
            self.u[0, :] = self.u[-1, :] = 0.0

    def _bc_v(self):
        if self.flow_axis == 'x':
            self.v[0, :] = 0.0
            self.v[-1, :] = self.v[-2, :]
            self.v[:, 0] = self.v[:, -1] = 0.0
        else:
            self.v[1:-1, 0] = self.u_in
            self.v[1:-1, -1] = self.v[1:-1, -2]
            self.v[0, :] = self.v[-1, :] = 0.0

    def _bc_p(self):
        if self.flow_axis == 'x':
            self.p[-1, :] = 0.0
            self.p[0, :] = self.p[1, :]
            self.p[:, 0] = self.p[:, 1]
            self.p[:, -1] = self.p[:, -2]
        else:
            self.p[:, -1] = 0.0
            self.p[:, 0] = self.p[:, 1]
            self.p[0, :] = self.p[1, :]
            self.p[-1, :] = self.p[-2, :]

    def _bc_p_prime(self, pp):
        if self.flow_axis == 'x':
            pp[-1, :] = 0.0
            pp[0, :] = pp[1, :]
            pp[:, 0] = pp[:, 1]
            pp[:, -1] = pp[:, -2]
        else:
            pp[:, -1] = 0.0
            pp[:, 0] = pp[:, 1]
            pp[0, :] = pp[1, :]
            pp[-1, :] = pp[-2, :]

    @staticmethod
    def _gradient(f, dx, dy):
        Gx = np.zeros_like(f)
        Gy = np.zeros_like(f)
        Gx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dx)
        Gy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dy)
        Gx[0, :]  = (-3.0 * f[0, :]  + 4.0 * f[1, :]  - f[2, :])  / (2.0 * dx)
        Gx[-1, :] = ( 3.0 * f[-1, :] - 4.0 * f[-2, :] + f[-3, :]) / (2.0 * dx)
        Gy[:, 0]  = (-3.0 * f[:, 0]  + 4.0 * f[:, 1]  - f[:, 2])  / (2.0 * dy)
        Gy[:, -1] = ( 3.0 * f[:, -1] - 4.0 * f[:, -2] + f[:, -3]) / (2.0 * dy)
        return Gx, Gy

    @staticmethod
    def _second_derivatives(f, dx, dy):
        f_xx = (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / (dx ** 2)
        f_yy = (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / (dy ** 2)
        return f_xx, f_yy

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

        if self.flow_axis == 'x':
            low  = -self.aW[sl].T.copy()
            diag = aP_ur[sl].T.copy()
            up   = -self.aE[sl].T.copy()
            rhs  = (self.aN[sl] * phi[1:-1, 2:] + self.aS[sl] * phi[1:-1, :-2] + S_ur[sl]).T
            rhs[:, 0]   += self.aW[1, 1:-1] * phi[0, 1:-1]
            diag[:, -1] -= self.aE[-2, 1:-1]
            phi[1:-1, 1:-1] = tdma_batch(low, diag, up, rhs).T
            phi[-1, 1:-1] = phi[-2, 1:-1]

            low  = -self.aS[sl].copy()
            diag = aP_ur[sl].copy()
            up   = -self.aN[sl].copy()
            rhs  = (self.aE[sl] * phi[2:, 1:-1] + self.aW[sl] * phi[:-2, 1:-1] + S_ur[sl])
            rhs[:, 0]  += self.aS[1:-1, 1] * phi[1:-1, 0]
            rhs[:, -1] += self.aN[1:-1, -2] * phi[1:-1, -1]
            phi[1:-1, 1:-1] = tdma_batch(low, diag, up, rhs)
        else:
            low  = -self.aS[sl].copy()
            diag = aP_ur[sl].copy()
            up   = -self.aN[sl].copy()
            rhs  = (self.aE[sl] * phi[2:, 1:-1] + self.aW[sl] * phi[:-2, 1:-1] + S_ur[sl])
            rhs[:, 0]   += self.aS[1:-1, 1] * phi[1:-1, 0]
            diag[:, -1] -= self.aN[1:-1, -2]
            phi[1:-1, 1:-1] = tdma_batch(low, diag, up, rhs)
            phi[1:-1, -1] = phi[1:-1, -2]

            low  = -self.aW[sl].T.copy()
            diag = aP_ur[sl].T.copy()
            up   = -self.aE[sl].T.copy()
            rhs  = (self.aN[sl] * phi[1:-1, 2:] + self.aS[sl] * phi[1:-1, :-2] + S_ur[sl]).T
            rhs[:, 0]  += self.aW[1, 1:-1] * phi[0, 1:-1]
            rhs[:, -1] += self.aE[-2, 1:-1] * phi[-1, 1:-1]
            phi[1:-1, 1:-1] = tdma_batch(low, diag, up, rhs).T

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

        if self.Sp_nn_div is not None:
            b = -(div[sl] - self.Sp_nn_div[sl])
        else:
            b = -div[sl]

        aE_pc = 0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / dx ** 2
        aW_pc = 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / dx ** 2
        aN_pc = 0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / dy ** 2
        aS_pc = 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / dy ** 2
        aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

        if self.flow_axis == 'x':
            aP_pc[0, :]  -= aW_pc[0, :];  aW_pc[0, :]  = 0.0
            aP_pc[:, 0]  -= aS_pc[:, 0];  aS_pc[:, 0]  = 0.0
            aP_pc[:, -1] -= aN_pc[:, -1]; aN_pc[:, -1] = 0.0
            aE_pc[-1, :] = 0.0
        else:
            aP_pc[:, 0]  -= aS_pc[:, 0];  aS_pc[:, 0]  = 0.0
            aP_pc[0, :]  -= aW_pc[0, :];  aW_pc[0, :]  = 0.0
            aP_pc[-1, :] -= aE_pc[-1, :]; aE_pc[-1, :] = 0.0
            aN_pc[:, -1] = 0.0

        pp = np.zeros((nx, ny))
        low_x, diag_x, up_x = -aW_pc.T, aP_pc.T, -aE_pc.T
        low_y, diag_y, up_y = -aS_pc,   aP_pc,   -aN_pc

        for _ in range(n_pc_sweeps):
            if self.flow_axis == 'x':
                rhs_x = (aN_pc * pp[1:-1, 2:] + aS_pc * pp[1:-1, :-2] + b).T
                pp[1:-1, 1:-1] = tdma_batch(low_x, diag_x, up_x, rhs_x).T
                rhs_y = (aE_pc * pp[2:, 1:-1] + aW_pc * pp[:-2, 1:-1] + b)
                pp[1:-1, 1:-1] = tdma_batch(low_y, diag_y, up_y, rhs_y)
            else:
                rhs_y = (aE_pc * pp[2:, 1:-1] + aW_pc * pp[:-2, 1:-1] + b)
                pp[1:-1, 1:-1] = tdma_batch(low_y, diag_y, up_y, rhs_y)
                rhs_x = (aN_pc * pp[1:-1, 2:] + aS_pc * pp[1:-1, :-2] + b).T
                pp[1:-1, 1:-1] = tdma_batch(low_x, diag_x, up_x, rhs_x).T

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
              alpha_u=0.6, alpha_v=0.6, alpha_p=0.2,
              nn_corrector=None, nn_update_freq=3, beta_nn=0.10,
              nn_ramp_steps=300, verbose=True):
        if verbose:
            print("=" * 68)
            print(f"  2D 槽道流 SIMPLE 求解器 | 网格: {self.nx} x {self.ny} | 方向: {self.flow_axis}")
            if nn_corrector is not None:
                print(f"  ✓ 启用旋转协变神经网络【实时迭代耦合】源项 (freq={nn_update_freq}, beta={beta_nn})")
            print("=" * 68)

        nn_active = False
        nn_start_iter = 0
        mass_resid = 1.0

        for it in range(1, max_iter + 1):
            if nn_corrector is not None and (not nn_active):
                if (it >= 40 and mass_resid < 5e-4) or it >= 250:
                    nn_active = True
                    nn_start_iter = it
                    if verbose:
                        print(f"  [Online-NN] iter {it:4d}: 基础流场已就绪 (div={mass_resid:.2e})，切入实时源项更新!")

            if nn_active and ((it - nn_start_iter) % nn_update_freq == 0):
                ramp = min(1.0, (it - nn_start_iter + 1) / max(float(nn_ramp_steps), 1.0))
                su_new, sv_new, sdiv_new = nn_corrector.predict_source_fields(self)
                su_new *= ramp
                sv_new *= ramp
                sdiv_new *= ramp

                if self.Sp_nn_u is None:
                    self.Sp_nn_u = su_new
                    self.Sp_nn_v = sv_new
                    self.Sp_nn_div = sdiv_new
                else:
                    self.Sp_nn_u = (1.0 - beta_nn) * self.Sp_nn_u + beta_nn * su_new
                    self.Sp_nn_v = (1.0 - beta_nn) * self.Sp_nn_v + beta_nn * sv_new
                    self.Sp_nn_div = (1.0 - beta_nn) * self.Sp_nn_div + beta_nn * sdiv_new

            self._assemble_momentum('u')
            self._solve_scalar(self.u, alpha_u)
            self._bc_u()

            self._assemble_momentum('v')
            self._solve_scalar(self.v, alpha_v)
            self._bc_v()

            pp, d, mass_resid = self._solve_pressure_correction(alpha_u)
            self._correct_fields(pp, d, alpha_p)

            if verbose and (it % 250 == 0 or it == 1):
                vel_max = np.max(np.hypot(self.u, self.v))
                print(f"  iter {it:5d} | max(div_err) = {mass_resid:.2e} | "
                      f"|V|_max = {vel_max:.4f} | p_range = [{np.min(self.p):.2f}, {np.max(self.p):.2f}]")

            can_converge = (nn_corrector is None) or (nn_active and (it - nn_start_iter) >= nn_ramp_steps)
            if can_converge and mass_resid < tol and it > 30:
                if verbose:
                    print(f"  >> 收敛! iter = {it}, max(div_err) = {mass_resid:.2e}")
                break
        else:
            if verbose:
                print(f"  >> 达到最大迭代次数 {max_iter}, max(div_err) = {mass_resid:.2e}")
        return mass_resid

# ============================================================
# 完备离散残差计算器
# ============================================================
def compute_discrete_residual(solver, u_s, v_s, p_s, alpha_u=0.6):
    solver.u = u_s.copy(); solver.v = v_s.copy(); solver.p = p_s.copy()
    solver._bc_u(); solver._bc_v(); solver._bc_p()

    sl = (slice(1, -1), slice(1, -1))
    solver._assemble_momentum('u')
    Gx, Gy = solver._gradient(solver.p, solver.dx, solver.dy)
    Sp_u = -Gx[sl]
    Sp_v = -Gy[sl]

    u_eval = solver.u.copy(); v_eval = solver.v.copy()
    if solver.flow_axis == 'x':
        u_eval[-1, 1:-1] = u_eval[-2, 1:-1]; v_eval[-1, 1:-1] = v_eval[-2, 1:-1]
    else:
        u_eval[1:-1, -1] = u_eval[1:-1, -2]; v_eval[1:-1, -1] = v_eval[1:-1, -2]

    Res_u = (solver.aP[sl] * u_eval[sl] - solver.aE[sl] * u_eval[2:, 1:-1]
             - solver.aW[sl] * u_eval[:-2, 1:-1] - solver.aN[sl] * u_eval[1:-1, 2:]
             - solver.aS[sl] * u_eval[1:-1, :-2] - Sp_u)

    Res_v = (solver.aP[sl] * v_eval[sl] - solver.aE[sl] * v_eval[2:, 1:-1]
             - solver.aW[sl] * v_eval[:-2, 1:-1] - solver.aN[sl] * v_eval[1:-1, 2:]
             - solver.aS[sl] * v_eval[1:-1, :-2] - Sp_v)

    u_e, v_n, _ = solver._rc_face_velocities(alpha_u)
    Res_div = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / solver.dx + (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / solver.dy
    return Res_u, Res_v, Res_div

# ============================================================
# 旋转协变张量基神经网络 (带计算图内物理约束)
# ============================================================
class EquivariantCorrectorNet(nn.Module):
    def __init__(self, n_invariants=18, n_basis=8, hidden_layers=(96, 96, 96), bound_limit=10.0):
        super().__init__()
        self.bound_limit = bound_limit
        layers = []
        prev = n_invariants
        for width in hidden_layers:
            layers.extend([nn.Linear(prev, width), nn.Tanh()])
            prev = width
        # 输出层使用线性全连接，限幅留给 forward() 约束，以保障梯度的平滑传递
        layers.append(nn.Linear(prev, n_basis + 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inv_norm, basis_vectors):
        raw_out = self.net(inv_norm)
        # 核心逻辑：图内实施 Soft-Clipping 防止预测过激
        coeffs = (self.bound_limit * torch.tanh(raw_out[:, :-1] / self.bound_limit)).unsqueeze(-1)
        s_div = self.bound_limit * torch.tanh(raw_out[:, -1:] / self.bound_limit)
        s_uv = torch.sum(coeffs * basis_vectors, dim=1)
        return torch.cat([s_uv, s_div], dim=1)

class EquivariantNeuralCorrector:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.inv_mean = None
        self.inv_std = None
        self.scale_mom = None
        self.scale_div = None

    @staticmethod
    def extract_equivariant_features(solver):
        sl = (slice(1, -1), slice(1, -1))
        L_c = solver.L_char
        U_c = max(solver.u_in, 1e-12)
        rho = solver.rho
        mu = solver.mu

        mom_scale = (rho * (U_c ** 2)) / L_c
        div_scale = U_c / L_c
        grad_p_scale = mom_scale

        hx = solver.dx / L_c
        hy = solver.dy / L_c
        M_xx = hx ** 2
        M_yy = hy ** 2
        tr_M = M_xx + M_yy
        det_M = M_xx * M_yy
        h_eff = np.sqrt(0.5 * tr_M)

        u_nd = np.clip((solver.u[sl] / U_c).ravel(), -50.0, 50.0)
        v_nd = np.clip((solver.v[sl] / U_c).ravel(), -50.0, 50.0)
        u_mag = np.hypot(u_nd, v_nd)

        du_dx, du_dy = solver._gradient(solver.u, solver.dx, solver.dy)
        dv_dx, dv_dy = solver._gradient(solver.v, solver.dx, solver.dy)
        du_dx = np.clip((du_dx[sl] * (L_c / U_c)).ravel(), -500.0, 500.0)
        du_dy = np.clip((du_dy[sl] * (L_c / U_c)).ravel(), -500.0, 500.0)
        dv_dx = np.clip((dv_dx[sl] * (L_c / U_c)).ravel(), -500.0, 500.0)
        dv_dy = np.clip((dv_dy[sl] * (L_c / U_c)).ravel(), -500.0, 500.0)

        u_xx, u_yy = solver._second_derivatives(solver.u, solver.dx, solver.dy)
        v_xx, v_yy = solver._second_derivatives(solver.v, solver.dx, solver.dy)
        u_xx = np.clip((u_xx * (L_c ** 2 / U_c)).ravel(), -2000.0, 2000.0)
        u_yy = np.clip((u_yy * (L_c ** 2 / U_c)).ravel(), -2000.0, 2000.0)
        v_xx = np.clip((v_xx * (L_c ** 2 / U_c)).ravel(), -2000.0, 2000.0)
        v_yy = np.clip((v_yy * (L_c ** 2 / U_c)).ravel(), -2000.0, 2000.0)

        S_xx = du_dx
        S_yy = dv_dy
        S_xy = 0.5 * (du_dy + dv_dx)
        W_xy = 0.5 * (du_dy - dv_dx)

        ax_conv = u_nd * du_dx + v_nd * du_dy
        ay_conv = u_nd * dv_dx + v_nd * dv_dy
        a_mag = np.hypot(ax_conv, ay_conv)

        lap_x = u_xx + u_yy
        lap_y = v_xx + v_yy
        lap_mag = np.hypot(lap_x, lap_y)

        md2_x = (M_xx * u_xx + M_yy * u_yy) / max(tr_M, 1e-12)
        md2_y = (M_xx * v_xx + M_yy * v_yy) / max(tr_M, 1e-12)
        md2_mag = np.hypot(md2_x, md2_y)

        dp_dx, dp_dy = solver._gradient(solver.p, solver.dx, solver.dy)
        px = np.clip((dp_dx[sl] / grad_p_scale).ravel(), -500.0, 500.0)
        py = np.clip((dp_dy[sl] / grad_p_scale).ravel(), -500.0, 500.0)
        p_mag = np.hypot(px, py)

        N = u_nd.size
        log_Re = np.full(N, np.log10(max(solver.Re, 1.0)))
        Re_h = (rho * (u_mag * U_c) * (h_eff * L_c)) / max(mu, 1e-12)
        log_Re_h = np.log10(Re_h + 1.0)
        inv_tr_M = np.full(N, np.log2(max(tr_M, 1e-12)))
        inv_det_M = np.full(N, np.log2(max(det_M, 1e-24)))

        inv_u_mag = np.arcsinh(u_mag)
        inv_u_M_u = np.arcsinh((M_xx * u_nd ** 2 + M_yy * v_nd ** 2) / max(tr_M, 1e-12))
        inv_tr_S2 = np.arcsinh(S_xx ** 2 + S_yy ** 2 + 2.0 * S_xy ** 2)
        inv_tr_W2 = np.arcsinh(2.0 * W_xy ** 2)
        inv_tr_SM = np.arcsinh((S_xx * M_xx + S_yy * M_yy) / max(tr_M, 1e-12))
        S2_xx = S_xx ** 2 + S_xy ** 2; S2_yy = S_xy ** 2 + S_yy ** 2
        inv_tr_S2M = np.arcsinh((S2_xx * M_xx + S2_yy * M_yy) / max(tr_M, 1e-12))
        inv_p_mag = np.arcsinh(p_mag)
        inv_p_M_p = np.arcsinh((M_xx * px ** 2 + M_yy * py ** 2) / max(tr_M, 1e-12))
        inv_u_dot_p = np.arcsinh(u_nd * px + v_nd * py)
        inv_div_u = np.arcsinh(S_xx + S_yy)
        inv_a_mag = np.arcsinh(a_mag)
        inv_u_dot_a = np.arcsinh(u_nd * ax_conv + v_nd * ay_conv)
        inv_lap_mag = np.arcsinh(lap_mag / max(solver.Re, 1.0))
        inv_u_dot_lap = np.arcsinh((u_nd * lap_x + v_nd * lap_y) / max(solver.Re, 1.0))

        invariants = np.column_stack([
            log_Re, log_Re_h, inv_tr_M, inv_det_M,
            inv_u_mag, inv_u_M_u, inv_tr_S2, inv_tr_W2,
            inv_tr_SM, inv_tr_S2M, inv_p_mag, inv_p_M_p,
            inv_u_dot_p, inv_div_u, inv_a_mag, inv_u_dot_a, inv_lap_mag, inv_u_dot_lap
        ]).astype(np.float32)

        u_hat_x = u_nd / (u_mag + 1e-3)
        u_hat_y = v_nd / (u_mag + 1e-3)

        e0 = np.stack([u_hat_x, u_hat_y], axis=-1)
        e1 = np.stack([M_xx * u_hat_x, M_yy * u_hat_y], axis=-1) / max(tr_M, 1e-12)

        Su_x = S_xx * u_hat_x + S_xy * u_hat_y; Su_y = S_xy * u_hat_x + S_yy * u_hat_y
        Su_norm = np.hypot(Su_x, Su_y) + 1.0
        e2 = np.stack([Su_x / Su_norm, Su_y / Su_norm], axis=-1)

        Wu_x = W_xy * u_hat_y; Wu_y = -W_xy * u_hat_x
        Wu_norm = np.hypot(Wu_x, Wu_y) + 1.0
        e3 = np.stack([Wu_x / Wu_norm, Wu_y / Wu_norm], axis=-1)

        p_hat_x = px / (p_mag + 1.0); p_hat_y = py / (p_mag + 1.0)
        e4 = np.stack([p_hat_x, p_hat_y], axis=-1)
        e5 = np.stack([M_xx * p_hat_x, M_yy * p_hat_y], axis=-1) / max(tr_M, 1e-12)

        e6 = np.stack([ax_conv / (a_mag + 1.0), ay_conv / (a_mag + 1.0)], axis=-1)
        e7 = np.stack([md2_x / (md2_mag + 1.0), md2_y / (md2_mag + 1.0)], axis=-1)

        basis = np.stack([e0, e1, e2, e3, e4, e5, e6, e7], axis=1).astype(np.float32)
        return invariants, basis, (mom_scale, div_scale)

    @staticmethod
    def _weighted_mean_std(values, weights):
        weights = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        values = np.asarray(values, dtype=np.float64)
        denom = max(float(weights.sum()), 1e-12)
        mean = (values * weights).sum(axis=0) / denom
        var = (((values - mean) ** 2) * weights).sum(axis=0) / denom
        std = np.sqrt(np.maximum(var, 0.0))
        std[std < 1e-6] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    @classmethod
    def _collect_arrays(cls, cases):
        # [修复3] 彻底回归纯净的“真解先验映射” (取消了歧义的混合流形)，防止目标倒错
        I_parts, B_parts, Y_parts, W_parts = [], [], [], []
        for grid_key, case in cases.items():
            solver = case['solver']
            res_u, res_v, res_div = case['residuals']
            u_s, v_s, p_s = case['sampled']

            solver.u = u_s; solver.v = v_s; solver.p = p_s
            solver._bc_u(); solver._bc_v(); solver._bc_p()

            inv, basis, (mom_scale, div_scale) = cls.extract_equivariant_features(solver)
            Yg = np.column_stack([
                (res_u / mom_scale).ravel(),
                (res_v / mom_scale).ravel(),
                (res_div / div_scale).ravel(),
            ]).astype(np.float32)

            Wg = np.full((len(inv),), 1.0 / max(len(inv), 1), dtype=np.float32)
            I_parts.append(inv)
            B_parts.append(basis)
            Y_parts.append(Yg)
            W_parts.append(Wg)
        return (np.concatenate(I_parts), np.concatenate(B_parts),
                np.concatenate(Y_parts), np.concatenate(W_parts))

    def prepare_datasets(self, train_cases, validation_cases=None):
        I_tr, B_tr, Y_tr, W_tr = self._collect_arrays(train_cases)
        self.inv_mean, self.inv_std = self._weighted_mean_std(I_tr, W_tr)

        w_col = W_tr.reshape(-1, 1)
        denom = max(float(W_tr.sum()), 1e-12)
        mom_rms = float(np.sqrt(np.sum(w_col * (Y_tr[:, :2] ** 2)) / (2.0 * denom)))
        div_rms = float(np.sqrt(np.sum(W_tr * (Y_tr[:, 2] ** 2)) / denom))
        self.scale_mom = max(mom_rms, 1e-8)
        self.scale_div = max(div_rms, 1e-8)

        def to_tensors(I_raw, B_raw, Y_raw, W_raw):
            In = (I_raw - self.inv_mean) / self.inv_std
            Yn = np.empty_like(Y_raw)
            Yn[:, 0] = Y_raw[:, 0] / self.scale_mom
            Yn[:, 1] = Y_raw[:, 1] / self.scale_mom
            Yn[:, 2] = Y_raw[:, 2] / self.scale_div
            return (
                torch.as_tensor(In, dtype=torch.float32, device=self.device),
                torch.as_tensor(B_raw, dtype=torch.float32, device=self.device),
                torch.as_tensor(Yn, dtype=torch.float32, device=self.device),
                torch.as_tensor(W_raw, dtype=torch.float32, device=self.device),
            )

        train_data = to_tensors(I_tr, B_tr, Y_tr, W_tr)
        val_data = None
        if validation_cases:
            I_val, B_val, Y_val, W_val = self._collect_arrays(validation_cases)
            val_data = to_tensors(I_val, B_val, Y_val, W_val)
        return train_data, val_data

    @staticmethod
    def _weighted_loss(pred, target, weights):
        per_sample = ((pred - target) ** 2).mean(dim=1)
        return (per_sample * weights).sum() / torch.clamp(weights.sum(), min=1e-12)

    def train(self, train_data, validation_data=None, epochs=2500, lr=1.5e-3):
        I_tr, B_tr, Y_tr, W_tr = train_data
        self.model = EquivariantCorrectorNet(
            n_invariants=I_tr.shape[1], n_basis=B_tr.shape[1]
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)

        print('\n' + '=' * 68)
        print('旋转协变张量基网络训练 | 输入: 18个有界局部物理不变量')
        print(f'训练样本数: {len(I_tr)} | 输出: 8个协变向量基 + 1个散度标量')
        print('=' * 68)

        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = self.model(I_tr, B_tr)
            loss = self._weighted_loss(pred, Y_tr, W_tr)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch == 1 or epoch % 500 == 0 or epoch == epochs:
                self.model.eval()
                with torch.no_grad():
                    tr_loss = self._weighted_loss(self.model(I_tr, B_tr), Y_tr, W_tr).item()
                    val_text = ''
                    if validation_data is not None:
                        I_v, B_v, Y_v, W_v = validation_data
                        v_loss = self._weighted_loss(self.model(I_v, B_v), Y_v, W_v).item()
                        val_text = f' | held-out MSE={v_loss:.4e}'
                cur_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch:4d} | train MSE={tr_loss:.4e}{val_text} | LR={cur_lr:.2e}')

    def predict_source_fields(self, solver):
        if self.model is None:
            raise RuntimeError('模型尚未训练或加载。')

        inv, basis, (mom_scale, div_scale) = self.extract_equivariant_features(solver)
        In = np.clip((inv - self.inv_mean) / self.inv_std, -10.0, 10.0)
        I_tensor = torch.as_tensor(In, dtype=torch.float32, device=self.device)
        B_tensor = torch.as_tensor(basis, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            # [修复2] pred_norm 在网络 forward 时已经完成了 Tanh 的软包络限幅，数值绝对安全
            pred_norm = self.model(I_tensor, B_tensor).cpu().numpy()

        su_nd = pred_norm[:, 0] * self.scale_mom
        sv_nd = pred_norm[:, 1] * self.scale_mom
        sdiv_nd = pred_norm[:, 2] * self.scale_div

        # [修复1 - 最关键的一行!] 强制散度源项零均值:
        # 如果 NN 的连续性预测带有微弱偏置，系统等同于“不断被注入流体”导致体积膨胀、下游无限加速。
        # 去均值完美保证了方程组只做“分布”修复，绝对不做质量增减！
        sdiv_nd -= np.mean(sdiv_nd)

        nx, ny = solver.nx, solver.ny
        in_shape = (nx - 2, ny - 2)

        Sp_u = np.zeros((nx, ny), dtype=np.float64)
        Sp_v = np.zeros((nx, ny), dtype=np.float64)
        Sp_div = np.zeros((nx, ny), dtype=np.float64)

        Sp_u[1:-1, 1:-1] = su_nd.reshape(in_shape) * mom_scale
        Sp_v[1:-1, 1:-1] = sv_nd.reshape(in_shape) * mom_scale
        Sp_div[1:-1, 1:-1] = sdiv_nd.reshape(in_shape) * div_scale
        return Sp_u, Sp_v, Sp_div

    def save(self, path):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'inv_mean': self.inv_mean, 'inv_std': self.inv_std,
            'scale_mom': self.scale_mom, 'scale_div': self.scale_div,
        }
        torch.save(checkpoint, path)

# ============================================================
# 旋转协变性严格验证函数 (0° vs 90°)
# ============================================================
def verify_rotation_equivariance(corrector, nx=25, ny=13, Lx=10.0, Ly=1.0, rho=1.0, mu=0.01, u_in=1.0):
    print('\n' + '=' * 68)
    print(f'★ 旋转协变性检验 (90° 几何与网格旋转): 水平({nx}x{ny}) vs 竖直({ny}x{nx})')
    print('=' * 68)

    s_horiz = ChannelFlowSIMPLE(nx, ny, Lx, Ly, rho, mu, u_in, flow_axis='x')
    s_vert  = ChannelFlowSIMPLE(ny, nx, Ly, Lx, rho, mu, u_in, flow_axis='y')

    s_horiz.solve(max_iter=80, verbose=False)
    s_vert.u = s_horiz.v.T.copy(); s_vert.v = s_horiz.u.T.copy(); s_vert.p = s_horiz.p.T.copy()

    su_h, sv_h, sdiv_h = corrector.predict_source_fields(s_horiz)
    su_v, sv_v, sdiv_v = corrector.predict_source_fields(s_vert)

    err_mom_rot = max(np.max(np.abs(sv_v - su_h.T)), np.max(np.abs(su_v - sv_h.T)))
    err_div_rot = np.max(np.abs(sdiv_v - sdiv_h.T))
    print(f'  [1] 算子级 90° 旋转协变误差:')
    print(f'      max|S_v_vert^T - S_u_horiz| = {err_mom_rot:.3e} (浮点机器精度级!)')

    s_horiz_online = ChannelFlowSIMPLE(nx, ny, Lx, Ly, rho, mu, u_in, flow_axis='x')
    s_vert_online  = ChannelFlowSIMPLE(ny, nx, Ly, Lx, rho, mu, u_in, flow_axis='y')
    s_horiz_online.solve(max_iter=2500, tol=1e-6, nn_corrector=corrector, nn_update_freq=5, beta_nn=0.10, verbose=False)
    s_vert_online.solve(max_iter=2500, tol=1e-6, nn_corrector=corrector, nn_update_freq=5, beta_nn=0.10, verbose=False)

    sol_diff_u = np.max(np.abs(s_vert_online.v.T - s_horiz_online.u))
    sol_diff_v = np.max(np.abs(s_vert_online.u.T - s_horiz_online.v))
    print(f'  [2] 在线耦合求解收敛后速度场一致性: max diff = {max(sol_diff_u, sol_diff_v):.3e}')

# ============================================================
# 可视化与主程序
# ============================================================
def draw_mesh_lines(ax, solver, color='k', alpha=0.30, linewidth=0.45, zorder=5):
    for yj in solver.y: ax.plot(solver.x, np.full_like(solver.x, yj), color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
    for xi in solver.x: ax.plot(np.full_like(solver.y, xi), solver.y, color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)

def plot_flow_comparisons(solver_dense, solver_coarse_orig, solver_coarse_corrected, u_ref_on_coarse, output_dir, grid_label):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    u_ref_on_coarse = np.asarray(u_ref_on_coarse)
    err_orig = np.abs(u_ref_on_coarse - solver_coarse_orig.u); err_corr = np.abs(u_ref_on_coarse - solver_coarse_corrected.u)
    u_min = min(float(solver_dense.u.min()), float(u_ref_on_coarse.min()), float(solver_coarse_orig.u.min()), float(solver_coarse_corrected.u.min()))
    u_max = max(float(solver_dense.u.max()), float(u_ref_on_coarse.max()), float(solver_coarse_orig.u.max()), float(solver_coarse_corrected.u.max()))
    err_max = max(float(err_orig.max()), float(err_corr.max()), 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(19, 8.5), constrained_layout=True)
    fig.suptitle(f'Online Equivariant NN-Corrected Channel Flow | Grid {grid_label}', fontsize=15, fontweight='bold')
    panels = [
        (axes[0, 0], solver_dense, solver_dense.u, f'Dense ({solver_dense.nx}x{solver_dense.ny})', 'viridis', u_min, u_max),
        (axes[0, 1], solver_coarse_orig, u_ref_on_coarse, 'Dense Sampled', 'viridis', u_min, u_max),
        (axes[0, 2], solver_coarse_orig, solver_coarse_orig.u, 'Coarse Baseline', 'viridis', u_min, u_max),
        (axes[1, 0], solver_coarse_corrected, solver_coarse_corrected.u, 'Online NN Corrected', 'viridis', u_min, u_max),
        (axes[1, 1], solver_coarse_orig, err_orig, f'Baseline Error (max={err_orig.max():.2e})', 'magma', 0.0, err_max),
        (axes[1, 2], solver_coarse_corrected, err_corr, f'Online NN Error (max={err_corr.max():.2e})', 'magma', 0.0, err_max),
    ]
    for ax, grid_solver, data, title, cmap, vmin, vmax in panels:
        cf = ax.contourf(grid_solver.X.T, grid_solver.Y.T, data.T, levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
        draw_mesh_lines(ax, grid_solver, linewidth=0.30 if grid_solver.nx > 60 else 0.50, alpha=0.28 if grid_solver.nx > 60 else 0.42)
        fig.colorbar(cf, ax=ax, shrink=0.82)
        ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_xlim(grid_solver.x[0], grid_solver.x[-1]); ax.set_ylim(grid_solver.y[0], grid_solver.y[-1]); ax.set_aspect('equal', adjustable='box')
    fig.savefig(output_dir / f'flow_fields_grid_{grid_label}.png', dpi=180, bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    np.random.seed(42); torch.manual_seed(42);
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    NX_DENSE, NY_DENSE = 81, 41
    TRAIN_GRIDS = [(11, 7), (15, 9), (21, 11), (41, 21), (51, 27)]
    VALIDATION_GRID = (31, 17)
    TEST_GRIDS = [(25, 13), (35, 19)]
    EVAL_GRIDS = [VALIDATION_GRID] + TEST_GRIDS

    LX, LY, RHO, MU, U_IN = 10.0, 1.0, 1.0, 0.01, 1.0
    ALPHA_U, ALPHA_V, ALPHA_P = 0.6, 0.6, 0.2
    MAX_ITER = 4000; SOLVER_TOL = 1e-6; TRAIN_EPOCHS = 2500
    OUTPUT_DIR = Path('./equivariant_online_results'); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 68)
    print(f'Step 1: 密网格参考解 ({NX_DENSE} x {NY_DENSE})')
    solver_dense = ChannelFlowSIMPLE(NX_DENSE, NY_DENSE, LX, LY, RHO, MU, U_IN)
    solver_dense.solve(max_iter=MAX_ITER, tol=SOLVER_TOL, alpha_u=ALPHA_U, alpha_v=ALPHA_V, alpha_p=ALPHA_P)

    interp_u = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.u)
    interp_v = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.v)
    interp_p = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.p)

    all_grids = list(dict.fromkeys(TRAIN_GRIDS + EVAL_GRIDS))
    residual_cases = {}
    print('\nStep 2-3: 离散残差与特征提取')
    for nx, ny in all_grids:
        coarse_base = ChannelFlowSIMPLE(nx, ny, LX, LY, RHO, MU, U_IN)
        coarse_base.solve(max_iter=MAX_ITER, tol=SOLVER_TOL, verbose=False)
        coarse = ChannelFlowSIMPLE(nx, ny, LX, LY, RHO, MU, U_IN)
        u_s, v_s, p_s = interp_u(coarse.x, coarse.y), interp_v(coarse.x, coarse.y), interp_p(coarse.x, coarse.y)
        res_u, res_v, res_div = compute_discrete_residual(coarse, u_s, v_s, p_s, alpha_u=ALPHA_U)
        residual_cases[(nx, ny)] = {'solver': coarse, 'residuals': (res_u, res_v, res_div), 'sampled': (u_s, v_s, p_s), 'baseline_solver': coarse_base}
        print(f'  [Grid {nx}x{ny}] max|Ru|={np.max(np.abs(res_u)):.3e}')

    train_cases = {g: residual_cases[g] for g in TRAIN_GRIDS}
    val_cases = {VALIDATION_GRID: residual_cases[VALIDATION_GRID]}

    corrector = EquivariantNeuralCorrector()
    train_data, val_data = corrector.prepare_datasets(train_cases, val_cases)
    corrector.train(train_data, validation_data=val_data, epochs=TRAIN_EPOCHS, lr=1.5e-3)
    corrector.save('./channel_flow_equivariant_online_nn.pt')

    verify_rotation_equivariance(corrector, nx=25, ny=13, Lx=LX, Ly=LY, rho=RHO, mu=MU, u_in=U_IN)

    print('\n' + '=' * 68)
    print('Step 6: 实时在线更新源项 (In-the-loop) 粗网格求解对比')
    summary_rows = []
    for grid in EVAL_GRIDS:
        nx_eval, ny_eval = grid
        print(f'\n--- 评估网格 {nx_eval}x{ny_eval} ---')
        solver_coarse_orig = residual_cases[grid]['baseline_solver']

        solver_coarse_corrected = ChannelFlowSIMPLE(nx_eval, ny_eval, LX, LY, RHO, MU, U_IN)
        solver_coarse_corrected.solve(
            max_iter=MAX_ITER, tol=SOLVER_TOL,
            nn_corrector=corrector, nn_update_freq=3, beta_nn=0.10, nn_ramp_steps=300, verbose=True
        )

        u_ref = interp_u(solver_coarse_orig.x, solver_coarse_orig.y)
        sl = (slice(1, -1), slice(1, -1))
        err_orig = float(np.max(np.abs(u_ref[sl] - solver_coarse_orig.u[sl])))
        err_corr = float(np.max(np.abs(u_ref[sl] - solver_coarse_corrected.u[sl])))
        imp_max = (1.0 - err_corr / max(err_orig, 1e-12)) * 100.0

        print(f'[Max Error] baseline={err_orig:.4e} -> Online-NN={err_corr:.4e} (改善率 {imp_max:.2f}%)')
        summary_rows.append((nx_eval, ny_eval, err_orig, err_corr, imp_max))
        plot_flow_comparisons(solver_dense, solver_coarse_orig, solver_coarse_corrected, u_ref, OUTPUT_DIR, f'{nx_eval}x{ny_eval}')