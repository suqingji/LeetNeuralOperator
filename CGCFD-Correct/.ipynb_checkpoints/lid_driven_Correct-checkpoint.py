#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 顶盖驱动方腔流求解器 —— 基于离散残差的源项误差补偿 (v3 完整修复版)
======================================================================
修复要点:
  1. 增加连续性方程离散散度残差补偿 (Res_div)，实现动量+连续性双残差自洽 (Dual-Consistency)
  2. 采样场在计算残差前严格调用粗网格边界条件 (_bc_u, _bc_v, _bc_p)，消除边界压力梯度不一致
  3. 修复压力修正方程壁面 Neumann 边界条件符号错误 (+= 改为 -=)
  4. 修复中心线速度剖面绘图的数组切片维度颠倒问题 (u[mid_i, :] 与 v[:, mid_j])
  5. 移除 1e30 病态大数法，统一以内部节点 (1,1) 作为压力参考点并用 spsolve 精确求解
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RectBivariateSpline


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
# 主求解器（修复完备版，支持动量+连续性双源项注入）
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

        # ===== 离散残差补偿源项接口 =====
        self.Sp_nn_u = None    # u 动量方程残差源项
        self.Sp_nn_v = None    # v 动量方程残差源项
        self.Sp_nn_div = None  # 连续性方程 (Rhie-Chow 散度) 残差源项

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
        # 统一以第一个内部节点 (1, 1) 为参考压力零点
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

        # [修复 Bug #1]: 注入连续性方程的离散散度残差 Sp_nn_div
        if self.Sp_nn_div is not None:
            b = -(div[sl] - self.Sp_nn_div[sl]).copy()
        else:
            b = -div[sl].copy()

        aE_pc = 0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / dx ** 2
        aW_pc = 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / dx ** 2
        aN_pc = 0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / dy ** 2
        aS_pc = 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / dy ** 2
        aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

        # [修复 Bug #3]: 壁面 Neumann 边界条件必须从主对角线减去 (-=) 越界系数!
        aP_pc[0, :]  -= aW_pc[0, :]
        aP_pc[-1, :] -= aE_pc[-1, :]
        aP_pc[:, 0]  -= aS_pc[:, 0]
        aP_pc[:, -1] -= aN_pc[:, -1]

        # [修复 Bug #5]: 固定内部节点 (0,0) 即全局 (1,1) 为参考点 p'=0 (无 1e30 病态数)
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
            print("=" * 60)
            print(f"  2D 顶盖驱动方腔流 SIMPLE 求解器 | 网格: {self.nx} x {self.ny} | Re = {Re:.1f}")
            if self.Sp_nn_u is not None:
                print(f"  ✓ 启用完备残差源项补偿 (Res_u + Res_v + Res_div)")
            print("=" * 60)

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
# 离散残差计算器（v3 完备自洽版）
# ============================================================
def compute_discrete_residual(solver, u_field, v_field, p_field):
    """
    对给定采样流场 (u, v, p) 严格按照粗网格求解器的边界条件与离散格式，
    计算动量方程残差 (res_u, res_v) 与连续性散度残差 (res_div)。
    """
    # [修复 Bug #2]: 将采样场装入粗网格模板，并严格执行粗网格的 _bc_u, _bc_v, _bc_p!
    # 确保算残差时的边界压力/速度与粗网格迭代时的边界规则 100% 一致
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

    # 2. 计算动量残差: Res = aP*φ_P - (aE*φ_E + aW*φ_W + aN*φ_N + aS*φ_S) - Sp
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

    # 3. [修复 Bug #1]: 计算采样场在粗网格 Rhie-Chow 界面速度下的离散散度残差
    u_e, v_n, _ = solver._rc_face_velocities()
    res_div = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / solver.dx + \
              (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / solver.dy

    return res_u, res_v, res_div, solver.u.copy(), solver.v.copy(), solver.p.copy()


# ============================================================
# 可视化函数
# ============================================================
def plot_flow_field(solver_dense, solver_coarse_orig, solver_coarse_corrected):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Lid-Driven Cavity Flow Comparison (v3 Fixed)", fontsize=14, fontweight='bold')

    solvers = [solver_dense, solver_coarse_orig, solver_coarse_corrected]
    titles = ['Dense Grid (ref)', 'Coarse (original)', 'Coarse (residual-corrected)']

    for col, (sv, title) in enumerate(zip(solvers, titles)):
        ax = axes[0, col]
        vel_mag = np.sqrt(sv.u**2 + sv.v**2)
        cf = ax.contourf(sv.X.T, sv.Y.T, vel_mag.T, levels=30, cmap='viridis')
        plt.colorbar(cf, ax=ax, label='|V|')
        ax.set_title(f'{title}: Velocity Magnitude')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')

        ax = axes[1, col]
        psi = sv._streamfunc()
        levels = np.linspace(psi.min(), psi.max(), 30)
        cf = ax.contourf(sv.X.T, sv.Y.T, psi.T, levels=levels, cmap='Spectral_r')
        plt.colorbar(cf, ax=ax, label='ψ')
        ax.set_title(f'{title}: Streamlines (ψ_min={psi.min():.4f})')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('./lid_cavity_v3_flow_field.png', dpi=150, bbox_inches='tight')
    print("流场对比图已保存: ./lid_cavity_v3_flow_field.png")
    plt.show()


def plot_profile_and_error(solver_dense, solver_coarse_orig, solver_coarse_corrected,
                           u_sampled, v_sampled):
    mid_i_d = solver_dense.nx // 2
    mid_j_d = solver_dense.ny // 2
    mid_i_c = solver_coarse_orig.nx // 2
    mid_j_c = solver_coarse_orig.ny // 2

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle("Centerline Velocity Profiles (v3 Fixed)", fontsize=13, fontweight='bold')

    # [修复 Bug #4]: 垂直中心线 x=L/2 应取第一维 mid_i: u[mid_i, :]
    ax = axes1[0]
    ax.plot(solver_dense.u[mid_i_d, :], solver_dense.y, 'k-', lw=2, label='Dense (ref)')
    ax.plot(solver_coarse_orig.u[mid_i_c, :], solver_coarse_orig.y, 'r--o', ms=4, lw=1.5, label='Coarse (orig)')
    ax.plot(solver_coarse_corrected.u[mid_i_c, :], solver_coarse_corrected.y, 'b:s', ms=5, lw=2, label='Coarse (corrected)')
    ax.set_xlabel('u (m/s)'); ax.set_ylabel('y')
    ax.set_title('Vertical centerline (x=L/2): u vs y')
    ax.legend(); ax.grid(True, alpha=0.3)

    # [修复 Bug #4]: 水平中心线 y=L/2 应取第二维 mid_j: v[:, mid_j]
    ax = axes1[1]
    ax.plot(solver_dense.x, solver_dense.v[:, mid_j_d], 'k-', lw=2, label='Dense (ref)')
    ax.plot(solver_coarse_orig.x, solver_coarse_orig.v[:, mid_j_c], 'r--o', ms=4, lw=1.5, label='Coarse (orig)')
    ax.plot(solver_coarse_corrected.x, solver_coarse_corrected.v[:, mid_j_c], 'b:s', ms=5, lw=2, label='Coarse (corrected)')
    ax.set_xlabel('x'); ax.set_ylabel('v (m/s)')
    ax.set_title('Horizontal centerline (y=L/2): v vs x')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./lid_cavity_v3_centerline_profiles.png', dpi=150, bbox_inches='tight')
    print("中心线速度剖面图已保存: ./lid_cavity_v3_centerline_profiles.png")
    plt.show()

    # 误差统计
    err_u_orig = np.max(np.abs(u_sampled - solver_coarse_orig.u))
    err_u_corr = np.max(np.abs(u_sampled - solver_coarse_corrected.u))
    err_v_orig = np.max(np.abs(v_sampled - solver_coarse_orig.v))
    err_v_corr = np.max(np.abs(v_sampled - solver_coarse_corrected.v))

    imp_u = (1 - err_u_corr / err_u_orig) * 100 if err_u_orig > 0 else 0
    imp_v = (1 - err_v_corr / err_v_orig) * 100 if err_v_orig > 0 else 0

    print(f"\n  误差对比 (max abs error vs dense ref):")
    print(f"    u: orig = {err_u_orig:.6e} → corrected = {err_u_corr:.6e}  (改善 {imp_u:.2f}%)")
    print(f"    v: orig = {err_v_orig:.6e} → corrected = {err_v_corr:.6e}  (改善 {imp_v:.2f}%)")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(2)
    width = 0.35
    bars1 = ax2.bar(x_pos - width/2, [err_u_orig, err_v_orig], width, label='Original', color='#ff6b6b')
    bars2 = ax2.bar(x_pos + width/2, [err_u_corr, err_v_corr], width, label='Corrected', color='#4ecdc4')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['u-velocity', 'v-velocity'])
    ax2.set_ylabel('Max Error')
    ax2.set_title(f'Max Error Reduction: u={imp_u:.2f}%, v={imp_v:.2f}%')
    ax2.legend()
    ax2.set_yscale('log')
    for bar, val in zip(list(bars1)+list(bars2), [err_u_orig, err_v_orig, err_u_corr, err_v_corr]):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.1, f'{val:.2e}',
                 ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('./lid_cavity_v3_error_stats.png', dpi=150, bbox_inches='tight')
    print("误差统计图已保存: ./lid_cavity_v3_error_stats.png")
    plt.show()

    return err_u_orig, err_u_corr, err_v_orig, err_v_corr


def plot_source_terms(res_u, res_v, res_div, solver):
    nx, ny = solver.nx, solver.ny
    ru_full = np.zeros((nx, ny));  ru_full[1:-1, 1:-1] = res_u
    rv_full = np.zeros((nx, ny));  rv_full[1:-1, 1:-1] = res_v
    rd_full = np.zeros((nx, ny));  rd_full[1:-1, 1:-1] = res_div

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Discrete Residual Source Terms (Res_u, Res_v, Res_div)", fontsize=13, fontweight='bold')

    for ax, field, name in zip(axes, [ru_full, rv_full, rd_full],
                               ['Res_u (u-momentum)', 'Res_v (v-momentum)', 'Res_div (Continuity)']):
        cf = ax.contourf(solver.X.T, solver.Y.T, field.T, levels=30, cmap='RdBu_r')
        plt.colorbar(cf, ax=ax)
        ax.set_title(name)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('./lid_cavity_v3_source_terms.png', dpi=150, bbox_inches='tight')
    print("源项分布图已保存: ./lid_cavity_v3_source_terms.png")
    plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    NX_DENSE, NY_DENSE   = 51, 51
    NX_COARSE, NY_COARSE = 21, 21

    L = 1.0
    RHO = 1.0
    MU = 0.01            # Re = 100
    U_LID = 1.0

    # ===== Step 1: 密网格求解（参考解）=====
    print("\n" + "="*60)
    print("Step 1: 密网格求解 (参考解)")
    print("="*60)
    solver_dense = LidDrivenCavitySIMPLE(NX_DENSE, NY_DENSE, L, RHO, MU, U_LID)
    solver_dense.solve(max_iter=2000, tol=1e-7, alpha_u=0.7, alpha_v=0.7, alpha_p=0.3)

    # ===== Step 2: 在粗网格节点上采样密网格解 =====
    print("\n" + "="*60)
    print("Step 2: 在粗网格节点上采样密网格解")
    print("="*60)
    interp_u = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.u)
    interp_v = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.v)
    interp_p = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.p)

    solver_template = LidDrivenCavitySIMPLE(NX_COARSE, NY_COARSE, L, RHO, MU, U_LID)
    u_raw = interp_u(solver_template.x, solver_template.y)
    v_raw = interp_v(solver_template.x, solver_template.y)
    p_raw = interp_p(solver_template.x, solver_template.y)

    # ===== Step 3: 计算完备离散残差 (Res_u, Res_v, Res_div) =====
    print("\n" + "="*60)
    print("Step 3: 计算粗网格上的完备离散残差")
    print("="*60)
    res_u, res_v, res_div, u_sampled, v_sampled, p_sampled = compute_discrete_residual(
        solver_template, u_raw, v_raw, p_raw
    )
    print(f"  Res_u   范围: [{res_u.min():.4e}, {res_u.max():.4e}], mean|x|={np.mean(np.abs(res_u)):.4e}")
    print(f"  Res_v   范围: [{res_v.min():.4e}, {res_v.max():.4e}], mean|x|={np.mean(np.abs(res_v)):.4e}")
    print(f"  Res_div 范围: [{res_div.min():.4e}, {res_div.max():.4e}], mean|x|={np.mean(np.abs(res_div)):.4e}")

    plot_source_terms(res_u, res_v, res_div, solver_template)

    # ===== Step 4: 粗网格基准求解（无修正）=====
    print("\n" + "="*60)
    print("Step 4: 粗网格基准求解 (无修正)")
    print("="*60)
    solver_coarse_orig = LidDrivenCavitySIMPLE(NX_COARSE, NY_COARSE, L, RHO, MU, U_LID)
    solver_coarse_orig.solve(max_iter=2000, tol=1e-7, alpha_u=0.7, alpha_v=0.7, alpha_p=0.3)

    # ===== Step 5: 粗网格修正求解（注入完备残差源项）=====
    print("\n" + "="*60)
    print("Step 5: 粗网格修正求解 (注入 Res_u + Res_v + Res_div)")
    print("="*60)
    res_u_full   = np.zeros((NX_COARSE, NY_COARSE)); res_u_full[1:-1, 1:-1]   = res_u
    res_v_full   = np.zeros((NX_COARSE, NY_COARSE)); res_v_full[1:-1, 1:-1]   = res_v
    res_div_full = np.zeros((NX_COARSE, NY_COARSE)); res_div_full[1:-1, 1:-1] = res_div

    solver_coarse_corrected = LidDrivenCavitySIMPLE(NX_COARSE, NY_COARSE, L, RHO, MU, U_LID)
    solver_coarse_corrected.Sp_nn_u   = res_u_full
    solver_coarse_corrected.Sp_nn_v   = res_v_full
    solver_coarse_corrected.Sp_nn_div = res_div_full
    solver_coarse_corrected.solve(max_iter=2000, tol=1e-7, alpha_u=0.7, alpha_v=0.7, alpha_p=0.3)

    # ===== Step 6: 误差分析与可视化 =====
    print("\n" + "="*60)
    print("Step 6: 误差对比分析")
    print("="*60)
    err_u_orig, err_u_corr, err_v_orig, err_v_corr = plot_profile_and_error(
        solver_dense, solver_coarse_orig, solver_coarse_corrected, u_sampled, v_sampled
    )
    plot_flow_field(solver_dense, solver_coarse_orig, solver_coarse_corrected)