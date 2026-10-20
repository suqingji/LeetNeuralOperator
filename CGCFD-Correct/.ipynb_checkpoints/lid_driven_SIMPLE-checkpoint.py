#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 顶盖驱动方腔流求解器（Bug修复与向量化完整版）
================================================
方法: FDM + 同位网格(Collocated Grid) + SIMPLE算法 + Rhie-Chow动量插值
修复内容:
  1. 修复压力修正方程壁面 Neumann 边界条件符号错误 (+= 改为 -=)
  2. 修复中心线速度剖面绘图切片维度颠倒问题 (u[mid_i, :] 与 v[:, mid_j])
  3. 移除破坏矩阵条件数的 1e30 大数法, 改用精确规范的内部参考点固定法
  4. 统一动量亚松弛与 Rhie-Chow 系数 d = alpha_u / aP 的自洽性
  5. 内置 Ghia et al. (1982) Re=400 权威基准解对比验证
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# 五对角稀疏矩阵构建（向量化，coo_matrix 显式索引）
# ============================================================
def build_penta_matrix(aP, aE, aW, aN, aS):
    """
    由内部节点系数数组构建五对角稀疏矩阵 A*x = b。
    所有输入形状均为 (nix, niy) = (nx-2, ny-2)。
    索引映射: idx = i * niy + j   (C-order flatten)
    """
    nix, niy = aP.shape
    n = nix * niy

    ii = np.arange(nix)[:, None] * niy + np.arange(niy)[None, :]
    rows_list, cols_list, vals_list = [], [], []

    # 主对角元
    rows_list.append(ii.ravel())
    cols_list.append(ii.ravel())
    vals_list.append(aP.ravel())

    # 东邻居 (i+1, j): idx + niy
    rows_list.append(ii[:-1, :].ravel())
    cols_list.append((ii[:-1, :] + niy).ravel())
    vals_list.append(-aE[:-1, :].ravel())

    # 西邻居 (i-1, j): idx - niy
    rows_list.append(ii[1:, :].ravel())
    cols_list.append((ii[1:, :] - niy).ravel())
    vals_list.append(-aW[1:, :].ravel())

    # 北邻居 (i, j+1): idx + 1
    rows_list.append(ii[:, :-1].ravel())
    cols_list.append((ii[:, :-1] + 1).ravel())
    vals_list.append(-aN[:, :-1].ravel())

    # 南邻居 (i, j-1): idx - 1
    rows_list.append(ii[:, 1:].ravel())
    cols_list.append((ii[:, 1:] - 1).ravel())
    vals_list.append(-aS[:, 1:].ravel())

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)

    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


# ============================================================
# 主求解器
# ============================================================
class LidDrivenCavitySIMPLE:
    def __init__(self, nx=81, ny=81, L=1.0, rho=1.0, mu=0.0025, u_lid=1.0):
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
        self.u[0, :]  = 0.0         # 左壁静止 (含左上角点)
        self.u[-1, :] = 0.0         # 右壁静止 (含右上角点)
        self.u[:, 0]  = 0.0         # 下壁静止

    def _bc_v(self):
        self.v[0, :]  = 0.0
        self.v[-1, :] = 0.0
        self.v[:, 0]  = 0.0
        self.v[:, -1] = 0.0

    def _bc_p(self):
        # 减去第一个内部节点 p[1,1] 以固定参考压力点
        self.p -= self.p[1, 1]
        # 四壁面法向零梯度 dp/dn = 0
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

        # 一阶迎风对流项
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
        self.Sp[sl] = -Gx[sl] if direction == 'u' else -Gy[sl]

    def _solve_scalar(self, phi, alpha):
        sl = self.sl
        nx, ny = self.nx, self.ny
        niy = ny - 2
        nix = nx - 2

        phi_old = phi.copy()

        aP_ur = self.aP[sl] / alpha
        S_ur = self.Sp[sl] + (1.0 - alpha) / alpha * self.aP[sl] * phi_old[sl]

        rhs = S_ur.ravel().copy()

        # 向量化添加四壁 Dirichlet 边界移项贡献
        rhs[0:niy]                 += self.aW[1, 1:-1]      * phi[0, 1:-1]
        rhs[(nix - 1) * niy:]      += self.aE[nx - 2, 1:-1] * phi[nx - 1, 1:-1]
        rhs[0:nix * niy:niy]       += self.aS[1:-1, 1]      * phi[1:-1, 0]
        rhs[niy - 1:nix * niy:niy] += self.aN[1:-1, ny - 2] * phi[1:-1, ny - 1]

        A = build_penta_matrix(aP_ur, self.aE[sl], self.aW[sl],
                               self.aN[sl], self.aS[sl])

        phi_internal = spsolve(A, rhs)
        phi[sl] = phi_internal.reshape((nix, niy), order='C')

    def _rc_face_velocities(self, alpha_u):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        d = np.zeros((nx, ny))
        d[1:-1, 1:-1] = alpha_u / self.aP[1:-1, 1:-1]
        d[0, :]  = d[1, :]
        d[-1, :] = d[-2, :]
        d[:, 0]  = d[:, 1]
        d[:, -1] = d[:, -2]

        Gx, Gy = self._gradient(self.p, dx, dy)

        d_bar_e = 0.5 * (d[:-1, :] + d[1:, :])
        Gx_bar_e = 0.5 * (Gx[:-1, :] + Gx[1:, :])
        Gx_compact_e = (self.p[1:, :] - self.p[:-1, :]) / dx
        u_e = 0.5 * (self.u[:-1, :] + self.u[1:, :]) + d_bar_e * (Gx_bar_e - Gx_compact_e)
        # 强制左右壁面处法向速度严格为 0
        u_e[0, :] = 0.0
        u_e[-1, :] = 0.0

        d_bar_n = 0.5 * (d[:, :-1] + d[:, 1:])
        Gy_bar_n = 0.5 * (Gy[:, :-1] + Gy[:, 1:])
        Gy_compact_n = (self.p[:, 1:] - self.p[:, :-1]) / dy
        v_n = 0.5 * (self.v[:, :-1] + self.v[:, 1:]) + d_bar_n * (Gy_bar_n - Gy_compact_n)
        # 强制上下壁面处法向速度严格为 0
        v_n[:, 0] = 0.0
        v_n[:, -1] = 0.0

        return u_e, v_n, d

    def _solve_pressure_correction(self, alpha_u):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        niy = ny - 2
        nix = nx - 2

        u_e, v_n, d = self._rc_face_velocities(alpha_u)

        div = np.zeros((nx, ny))
        sl = self.sl
        div[sl] = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / dx + \
                  (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / dy
        b = -div[sl].copy()

        aE_pc = 0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / dx ** 2
        aW_pc = 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / dx ** 2
        aN_pc = 0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / dy ** 2
        aS_pc = 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / dy ** 2
        aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

        # [修复 Bug #1]: 壁面零梯度 Neumann BC 应从主对角线减去 (-=) 越界系数!
        aP_pc[0, :]  -= aW_pc[0, :]
        aP_pc[-1, :] -= aE_pc[-1, :]
        aP_pc[:, 0]  -= aS_pc[:, 0]
        aP_pc[:, -1] -= aN_pc[:, -1]

        # [修复 Bug #3]: 固定内部节点 (0,0) 为参考压力点 p'=0, 保持正常量级避免矩阵病态
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

        mass_resid = np.max(np.abs(div[sl]))
        return pp, d, mass_resid

    def _correct_fields(self, pp, d, alpha_p):
        dx, dy = self.dx, self.dy
        sl = self.sl

        self.p[sl] += alpha_p * pp[sl]

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

    def solve(self, max_iter=3000, tol=1e-6,
              alpha_u=0.7, alpha_v=0.7, alpha_p=0.3, verbose=True):
        Re = self.rho * self.u_lid * self.L / self.mu
        if verbose:
            print("=" * 65)
            print(f"  2D 顶盖驱动方腔流 SIMPLE 求解器 (修复与向量化版)")
            print(f"  网格: {self.nx} x {self.ny} | Re = {Re:.1f}")
            print("=" * 65)

        u_old, v_old = self.u.copy(), self.v.copy()

        for it in range(1, max_iter + 1):
            self._assemble_momentum('u')
            self._solve_scalar(self.u, alpha_u)
            self._bc_u()

            self._assemble_momentum('v')
            self._solve_scalar(self.v, alpha_v)
            self._bc_v()

            pp, d, mass_resid = self._solve_pressure_correction(alpha_u)
            self._correct_fields(pp, d, alpha_p)

            res_u = self._rel_change(self.u, u_old)
            res_v = self._rel_change(self.v, v_old)
            u_old, v_old = self.u.copy(), self.v.copy()

            if verbose and (it % 50 == 0 or it == 1):
                psi_min = np.min(self._streamfunc())
                print(f"  iter {it:4d} | res_u={res_u:.2e} | res_v={res_v:.2e} | "
                      f"max(div)={mass_resid:.2e} | psi_min={psi_min:.4f}")

            if res_u < tol and res_v < tol and mass_resid < tol * 10 and it > 10:
                if verbose:
                    print(f"  >> 收敛! iter = {it}, res_u = {res_u:.2e}, max(div) = {mass_resid:.2e}")
                break
        else:
            if verbose:
                print(f"  >> 达到最大迭代次数 {max_iter}")

        return res_u, res_v, mass_resid

    def _streamfunc(self):
        """计算流函数 ψ: u = ∂ψ/∂y"""
        psi = np.zeros((self.nx, self.ny))
        psi[:, 1:] = np.cumsum(0.5 * (self.u[:, 1:] + self.u[:, :-1]) * self.dy, axis=1)
        return psi

    def plot_results(self, save_path=None):
        fig = plt.figure(figsize=(15, 12))
        Re = self.rho * self.u_lid * self.L / self.mu
        fig.suptitle(f"Lid-Driven Cavity Flow (Re = {Re:.0f}, Grid = {self.nx}×{self.ny}) — SIMPLE",
                     fontsize=14, fontweight='bold')

        # --- 子图 1: 速度幅值云图 ---
        ax1 = fig.add_subplot(2, 2, 1)
        vel_mag = np.sqrt(self.u ** 2 + self.v ** 2)
        cf1 = ax1.contourf(self.X.T, self.Y.T, vel_mag.T, levels=30, cmap='viridis')
        plt.colorbar(cf1, ax=ax1, label='|V| (m/s)')
        ax1.set_xlabel('x'); ax1.set_ylabel('y')
        ax1.set_title('Velocity Magnitude')
        ax1.set_aspect('equal')

        # --- 子图 2: 压力云图 ---
        ax2 = fig.add_subplot(2, 2, 2)
        cf2 = ax2.contourf(self.X.T, self.Y.T, self.p.T, levels=30, cmap='coolwarm')
        plt.colorbar(cf2, ax=ax2, label='p (Pa)')
        ax2.set_xlabel('x'); ax2.set_ylabel('y')
        ax2.set_title('Pressure Field')
        ax2.set_aspect('equal')

        # --- 子图 3: 流函数与流线图 ---
        ax3 = fig.add_subplot(2, 2, 3)
        psi = self._streamfunc()
        cf3 = ax3.contourf(self.X.T, self.Y.T, psi.T, levels=30, cmap='Spectral_r')
        plt.colorbar(cf3, ax=ax3, label='ψ')
        ax3.streamplot(self.x, self.y, self.u.T, self.v.T, color='k', linewidth=0.7, density=1.4)
        ax3.set_xlabel('x'); ax3.set_ylabel('y')
        ax3.set_title(f'Streamlines (ψ_min = {psi.min():.4f})')
        ax3.set_aspect('equal')

        # --- 子图 4: 中心线速度剖面 vs Ghia et al. (1982) 基准解 ---
        ax4 = fig.add_subplot(2, 2, 4)
        mid_i = self.nx // 2
        mid_j = self.ny // 2

        # [修复 Bug #2]: 正确索引垂直中心线 u(x=L/2, y) 与水平中心线 v(x, y=L/2)
        ax4.plot(self.u[mid_i, :], self.y, 'b-', lw=2, label='u(x=0.5, y) Numerical')
        ax4.plot(self.x, self.v[:, mid_j], 'r--', lw=2, label='v(x, y=0.5) Numerical')

        # 若 Re == 400，叠加 Ghia et al. (1982) 经典基准数据点
        if abs(Re - 400.0) < 1.0:
            ghia_y = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                      0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]
            ghia_u = [0.0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119,
                      -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.0]
            ghia_x = [0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                      0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0]
            ghia_v = [0.0, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174,
                      -0.05186, -0.38598, -0.44993, -0.33827, -0.22847, -0.19254, -0.15663, -0.12146, 0.0]
            ax4.plot(ghia_u, ghia_y, 'bo', ms=5, fillstyle='none', label='u Ghia et al. (1982)')
            ax4.plot(ghia_x, ghia_v, 'rs', ms=5, fillstyle='none', label='v Ghia et al. (1982)')

        ax4.set_xlabel('u(0.5, y)  or  x')
        ax4.set_ylabel('y  or  v(x, 0.5)')
        ax4.set_title('Centerline Velocity Profiles')
        ax4.legend(loc='lower right', fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  图片已保存: {save_path}")

        plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    NX    = 201            # x 方向网格数
    NY    = 201            # y 方向网格数
    L     = 1.0           # 方腔边长 (m)
    RHO   = 1.0           # 密度 (kg/m^3)
    MU    = 0.0025        # 动力粘度 → Re = 400
    U_LID = 1.0           # 顶盖速度 (m/s)

    solver = LidDrivenCavitySIMPLE(
        nx=NX, ny=NY, L=L,
        rho=RHO, mu=MU, u_lid=U_LID
    )

    solver.solve(
        max_iter=1500,
        tol=1e-6,
        alpha_u=0.7,
        alpha_v=0.7,
        alpha_p=0.3,
    )

    solver.plot_results(save_path='./lid_driven_cavity_pure.png')