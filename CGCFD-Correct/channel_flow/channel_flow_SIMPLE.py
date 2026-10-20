#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 不可压缩槽道流求解器 (修复与向量化完整版)
=============================================
方法: FDM + 同位网格(Collocated Grid) + SIMPLE算法 + Rhie-Chow动量插值
方程: 稳态二维不可压 Navier-Stokes
网格: 结构化均匀网格, 所有变量存储在相同节点

计算域:  0 <= x <= Lx,  0 <= y <= Ly
边界条件:
  入口 (x=0):   u = u_in (均匀), v = 0, dp/dx = 0
  出口 (x=Lx):  du/dx = dv/dx = 0, p = 0 (参考压力)
  下壁面 (y=0):  u = v = 0 (无滑移), dp/dy = 0
  上壁面 (y=Ly): u = v = 0 (无滑移), dp/dy = 0
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 向量化 Thomas 算法 (同时求解多条三对角线)
# ============================================================
def tdma_batch(low, diag, up, rhs):
    """
    同时求解 N 条独立的三对角方程组.
    求解形式: low*x[k-1] + diag*x[k] + up*x[k+1] = rhs

    注意: 对于 CFD 离散方程 aP*phi = aE*phi_E + aW*phi_W + ...,
    移项后为 -aW*phi_W + aP*phi - aE*phi_E = ...,
    因此传入 TDMA 时 low=-aW, diag=aP, up=-aE.
    """
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
# 主求解器
# ============================================================
class ChannelFlowSIMPLE:
    def __init__(self, nx=51, ny=21, Lx=5.0, Ly=1.0,
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

        # 流场 (同位网格: u, v, p 共用同一套节点)
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.p = np.zeros((nx, ny))

        # 动量方程系数
        sh = (nx, ny)
        self.aE = np.zeros(sh); self.aW = np.zeros(sh)
        self.aN = np.zeros(sh); self.aS = np.zeros(sh)
        self.aP = np.zeros(sh); self.Sp = np.zeros(sh)

        self._init_fields()

    # ----------------------------------------------------------
    # 初始化与边界条件
    # ----------------------------------------------------------
    def _init_fields(self):
        self.u[:, :] = self.u_in
        self.v[:, :] = 0.0
        self.p[:, :] = 0.0
        self.u[:, 0] = self.u[:, -1] = 0.0   # 壁面无滑移
        self.u[0, 0] = self.u[-1, 0] = self.u[0, -1] = self.u[-1, -1] = 0.0

    def _bc_u(self):
        self.u[0, 1:-1] = self.u_in              # 入口
        self.u[-1, 1:-1] = self.u[-2, 1:-1]      # 出口零梯度
        self.u[:, 0] = self.u[:, -1] = 0.0       # 上下壁面无滑移

    def _bc_v(self):
        self.v[0, :] = 0.0                       # 入口
        self.v[-1, :] = self.v[-2, :]            # 出口零梯度
        self.v[:, 0] = self.v[:, -1] = 0.0       # 上下壁面无穿透

    def _bc_p(self):
        self.p[-1, :] = 0.0                      # 出口参考压力
        self.p[0, :] = self.p[1, :]              # 入口零梯度
        self.p[:, 0] = self.p[:, 1]              # 壁面零梯度
        self.p[:, -1] = self.p[:, -2]

    def _bc_p_prime(self, pp):
        """压力修正的边界条件"""
        pp[-1, :] = 0.0                          # 出口 p'=0
        pp[0, :] = pp[1, :]                      # 入口 dp'/dx=0
        pp[:, 0] = pp[:, 1]                      # 壁面 dp'/dn=0
        pp[:, -1] = pp[:, -2]

    # ----------------------------------------------------------
    # 压力梯度 (内部中心差分 + 边界二阶单侧差分)
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # 组装动量方程系数
    # ----------------------------------------------------------
    def _assemble_momentum(self, direction):
        """
        direction = 'u' or 'v'
        对流: 一阶迎风差分  扩散: 二阶中心差分
        所有 a-系数均为正数 (Patankar 约定).
        """
        dx, dy = self.dx, self.dy
        rho, mu = self.rho, self.mu
        sl = (slice(1, -1), slice(1, -1))

        ui = self.u[sl]
        vi = self.v[sl]

        De = Dw = mu / dx ** 2
        Dn = Ds = mu / dy ** 2

        # 迎风对流贡献 (系数为正)
        aE_c = np.where(ui >= 0, 0.0, -rho * ui / dx)
        aW_c = np.where(ui >= 0,  rho * ui / dx, 0.0)
        aN_c = np.where(vi >= 0, 0.0, -rho * vi / dy)
        aS_c = np.where(vi >= 0,  rho * vi / dy, 0.0)

        self.aE[sl] = De + aE_c
        self.aW[sl] = Dw + aW_c
        self.aN[sl] = Dn + aN_c
        self.aS[sl] = Ds + aS_c
        self.aP[sl] = self.aE[sl] + self.aW[sl] + self.aN[sl] + self.aS[sl]

        # 压力梯度源项 (-dp/dx 或 -dp/dy)
        Gx, Gy = self._gradient(self.p, dx, dy)
        self.Sp[sl] = -Gx[sl] if direction == 'u' else -Gy[sl]

    # ----------------------------------------------------------
    # 线迭代 TDMA 求解标量场 (已修复边界移项 Bug)
    # ----------------------------------------------------------
    def _solve_scalar(self, phi, alpha):
        """
        先 x 扫掠再 y 扫掠, 带隐式亚松弛.
        修复说明:
          1. 在 x 扫掠时, 显式将左边界(i=0)和右边界(i=nx-1)移项加入 rhs;
          2. 在 x 扫掠中隐式处理出口零梯度条件 (phi[-1] = phi[-2]).
        """
        sl = (slice(1, -1), slice(1, -1))
        phi_old = phi.copy()

        # 亚松弛嵌入方程: aP_ur = aP/alpha, S_ur = Sp + (1-alpha)/alpha * aP * phi_old
        aP_ur = self.aP.copy()
        S_ur  = self.Sp.copy()
        aP_ur[sl] = self.aP[sl] / alpha
        S_ur[sl]  = self.Sp[sl] + (1.0 - alpha) / alpha * self.aP[sl] * phi_old[sl]

        # --- x 方向扫掠 (固定 j, 沿 i=1..nx-2 求解) ---
        low  = -self.aW[sl].T.copy()                    # (ny-2, nx-2)
        diag = aP_ur[sl].T.copy()
        up   = -self.aE[sl].T.copy()
        rhs  = (self.aN[sl] * phi[1:-1, 2:] +
                self.aS[sl] * phi[1:-1, :-2] +
                S_ur[sl]).T

        # [修复 Bug #3]: 补上 x 扫掠左端入口 (i=0) 的已知边界贡献
        rhs[:, 0] += self.aW[1, 1:-1] * phi[0, 1:-1]
        # [修复 Bug #3]: 右端出口 (i=nx-1) 为零梯度边界 phi[-1,:] = phi[-2,:], 隐式并入主对角线
        diag[:, -1] -= self.aE[-2, 1:-1]

        phi[1:-1, 1:-1] = tdma_batch(low, diag, up, rhs).T
        phi[-1, 1:-1] = phi[-2, 1:-1]                   # 同步更新出口值供 y 扫掠使用

        # --- y 方向扫掠 (固定 i, 沿 j=1..ny-2 求解) ---
        low  = -self.aS[sl].copy()                      # (nx-2, ny-2)
        diag = aP_ur[sl].copy()
        up   = -self.aN[sl].copy()
        rhs  = (self.aE[sl] * phi[2:, 1:-1] +
                self.aW[sl] * phi[:-2, 1:-1] +
                S_ur[sl])

        # [修复 Bug #3]: 补上 y 扫掠上下壁面 (j=0 和 j=ny-1) 的边界贡献
        rhs[:, 0]  += self.aS[1:-1, 1] * phi[1:-1, 0]
        rhs[:, -1] += self.aN[1:-1, -2] * phi[1:-1, -1]

        phi[1:-1, 1:-1] = tdma_batch(low, diag, up, rhs)

    # ----------------------------------------------------------
    # Rhie-Chow 界面速度 (全向量化版本)
    # ----------------------------------------------------------
    def _rc_face_velocities(self, alpha_u):
        """
        计算 Rhie-Chow 插值的界面速度 u_e, v_n (无 Python 循环).
        u_e 位于 (i+1/2, j),  shape = (nx-1, ny)
        v_n 位于 (i, j+1/2),  shape = (nx, ny-1)
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        # d = alpha_u / aP (与亚松弛动量方程保持一致)
        d = np.zeros((nx, ny))
        d[1:-1, 1:-1] = alpha_u / self.aP[1:-1, 1:-1]
        d[0, :]  = d[1, :]
        d[-1, :] = d[-2, :]
        d[:, 0]  = d[:, 1]
        d[:, -1] = d[:, -2]

        Gx, Gy = self._gradient(self.p, dx, dy)

        # --- u 在东界面 (i+1/2, j), i = 0..nx-2 ---
        d_bar_e = 0.5 * (d[:-1, :] + d[1:, :])
        Gx_bar_e = 0.5 * (Gx[:-1, :] + Gx[1:, :])
        Gx_compact_e = (self.p[1:, :] - self.p[:-1, :]) / dx
        u_e = 0.5 * (self.u[:-1, :] + self.u[1:, :]) + d_bar_e * (Gx_bar_e - Gx_compact_e)

        # --- v 在北界面 (i, j+1/2), j = 0..ny-2 ---
        d_bar_n = 0.5 * (d[:, :-1] + d[:, 1:])
        Gy_bar_n = 0.5 * (Gy[:, :-1] + Gy[:, 1:])
        Gy_compact_n = (self.p[:, 1:] - self.p[:, :-1]) / dy
        v_n = 0.5 * (self.v[:, :-1] + self.v[:, 1:]) + d_bar_n * (Gy_bar_n - Gy_compact_n)

        return u_e, v_n, d

    # ----------------------------------------------------------
    # 压力修正方程 (修复边界符号、交叉重复累加及内迭代不足)
    # ----------------------------------------------------------
    def _solve_pressure_correction(self, alpha_u, n_pc_sweeps=25):
        """
        组装并求解压力修正 Poisson 方程.
        返回 p_prime, d 场和最大质量散度残差.
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        # Rhie-Chow 界面速度与 d 系数场
        u_e, v_n, d = self._rc_face_velocities(alpha_u)

        # 质量残差 (内部节点 i∈[1,nx-2], j∈[1,ny-2])
        div = np.zeros((nx, ny))
        sl = (slice(1, nx - 1), slice(1, ny - 1))
        div[sl] = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / dx + \
                  (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / dy
        b = -div[sl]       # 源项 = 负散度

        # 压力修正方程内部节点系数 (shape: nx-2, ny-2)
        aE_pc = 0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / dx ** 2
        aW_pc = 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / dx ** 2
        aN_pc = 0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / dy ** 2
        aS_pc = 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / dy ** 2
        aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

        # [修复 Bug #2]: 正确融入 Neumann (零梯度) 边界并清零越界邻接系数，防止交叉扫掠重复累加
        # 1. 入口 (i=1 邻接 i=0): dp'/dx=0 -> p'[0,:] = p'[1,:] -> aP 减去 aW, 且 aW 置零
        aP_pc[0, :] -= aW_pc[0, :]
        aW_pc[0, :] = 0.0

        # 2. 下壁面 (j=1 邻接 j=0): dp'/dy=0 -> p'[:,0] = p'[:,1] -> aP 减去 aS, 且 aS 置零
        aP_pc[:, 0] -= aS_pc[:, 0]
        aS_pc[:, 0] = 0.0

        # 3. 上壁面 (j=ny-2 邻接 j=ny-1): dp'/dy=0 -> p'[:,-1] = p'[:,-2] -> aP 减去 aN, 且 aN 置零
        aP_pc[:, -1] -= aN_pc[:, -1]
        aN_pc[:, -1] = 0.0

        # 4. 出口 (i=nx-2 邻接 i=nx-1): Dirichlet 边界 p'[-1,:]=0 -> aP 不变, 将 aE 置零
        aE_pc[-1, :] = 0.0

        # [修复 Bug #1]: 椭圆型方程需要多次交替方向 TDMA 内迭代以充分传递全局压力修正波
        pp = np.zeros((nx, ny))
        low_x, diag_x, up_x = -aW_pc.T, aP_pc.T, -aE_pc.T
        low_y, diag_y, up_y = -aS_pc,   aP_pc,   -aN_pc

        for _ in range(n_pc_sweeps):
            # x 方向扫掠
            rhs_x = (aN_pc * pp[1:-1, 2:] + aS_pc * pp[1:-1, :-2] + b).T
            pp[1:-1, 1:-1] = tdma_batch(low_x, diag_x, up_x, rhs_x).T

            # y 方向扫掠
            rhs_y = (aE_pc * pp[2:, 1:-1] + aW_pc * pp[:-2, 1:-1] + b)
            pp[1:-1, 1:-1] = tdma_batch(low_y, diag_y, up_y, rhs_y)

        self._bc_p_prime(pp)

        mass_resid = np.max(np.abs(div[sl]))
        return pp, d, mass_resid

    # ----------------------------------------------------------
    # 场修正
    # ----------------------------------------------------------
    def _correct_fields(self, pp, d, alpha_p):
        dx, dy = self.dx, self.dy
        sl = (slice(1, -1), slice(1, -1))

        # 压力修正
        self.p[sl] += alpha_p * pp[sl]

        # 速度修正 (使用节点的中央差分压力修正梯度)
        Gx_pp, Gy_pp = self._gradient(pp, dx, dy)
        self.u[sl] -= d[sl] * Gx_pp[sl]
        self.v[sl] -= d[sl] * Gy_pp[sl]

        self._bc_u()
        self._bc_v()
        self._bc_p()

    # ----------------------------------------------------------
    # 主迭代循环
    # ----------------------------------------------------------
    def solve(self, max_iter=4000, tol=1e-6,
              alpha_u=0.6, alpha_v=0.6, alpha_p=0.2):
        print("=" * 65)
        print(f"  2D 槽道流 SIMPLE 求解器 (Collocated FDM + Rhie-Chow)")
        print(f"  网格: {self.nx} x {self.ny}")
        print(f"  Re = {self.rho * self.u_in * self.Ly / self.mu:.1f}")
        print(f"  dx = {self.dx:.4f}, dy = {self.dy:.4f}")
        print("=" * 65)

        for it in range(1, max_iter + 1):
            # 1. 求解 u 动量方程
            self._assemble_momentum('u')
            self._solve_scalar(self.u, alpha_u)
            self._bc_u()

            # 2. 求解 v 动量方程
            self._assemble_momentum('v')
            self._solve_scalar(self.v, alpha_v)
            self._bc_v()

            # 3. 求解压力修正方程
            pp, d, mass_resid = self._solve_pressure_correction(alpha_u)

            # 4. 修正压力与速度场
            self._correct_fields(pp, d, alpha_p)

            if it % 100 == 0 or it == 1:
                u_max = np.max(self.u)
                print(f"  iter {it:5d} | max(div) = {mass_resid:.2e} | "
                      f"u_max = {u_max:.4f} | p_range = [{np.min(self.p):.2f}, {np.max(self.p):.2f}]")

            if mass_resid < tol and it > 10:
                print(f"  >> 收敛! iter = {it}, max(div) = {mass_resid:.2e}")
                break
        else:
            print(f"  >> 达到最大迭代次数 {max_iter}, max(div) = {mass_resid:.2e}")

        return mass_resid

    # ----------------------------------------------------------
    # 解析解 (充分发展泊肃叶流)
    # ----------------------------------------------------------
    def analytical_u(self):
        """
        充分发展段抛物线速度剖面.
        注: 使用梯形法则计算入口实际有效平均流速 u_mean,
            消除 FDM 网格在壁面角点 u=0 带来的离散流量折减 (ny-2)/(ny-1).
        """
        u_mean = np.trapezoid(self.u[0, :], self.y) / self.Ly if hasattr(np, 'trapezoid') else np.trapz(self.u[0, :], self.y) / self.Ly
        y_norm = self.y / self.Ly
        return 1.5 * u_mean * 4.0 * y_norm * (1.0 - y_norm)

    # ----------------------------------------------------------
    # 可视化
    # ----------------------------------------------------------
    def plot_results(self, save_path=None):
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("2D Incompressible Channel Flow — SIMPLE (Collocated FDM)",
                     fontsize=14, fontweight='bold')

        # --- 子图 1: 速度幅值云图 ---
        ax1 = fig.add_subplot(2, 2, 1)
        vel_mag = np.sqrt(self.u ** 2 + self.v ** 2)
        cf1 = ax1.contourf(self.X.T, self.Y.T, vel_mag.T, levels=30, cmap='viridis')
        plt.colorbar(cf1, ax=ax1, label='|V| (m/s)')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title('Velocity Magnitude')
        ax1.set_aspect('equal')

        # --- 子图 2: 压力云图 ---
        ax2 = fig.add_subplot(2, 2, 2)
        cf2 = ax2.contourf(self.X.T, self.Y.T, self.p.T, levels=30, cmap='coolwarm')
        plt.colorbar(cf2, ax=ax2, label='p (Pa)')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_title('Pressure Field')
        ax2.set_aspect('equal')

        # --- 子图 3: 速度矢量 ---
        ax3 = fig.add_subplot(2, 2, 3)
        step_x = max(1, self.nx // 25)
        step_y = max(1, self.ny // 10)
        ax3.quiver(self.X.T[::step_y, ::step_x], self.Y.T[::step_y, ::step_x],
                   self.u.T[::step_y, ::step_x], self.v.T[::step_y, ::step_x],
                   color='steelblue', alpha=0.8)
        ax3.set_xlabel('x (m)')
        ax3.set_ylabel('y (m)')
        ax3.set_title('Velocity Vectors')
        ax3.set_aspect('equal')

        # --- 子图 4: 截面速度剖面 vs 解析解 ---
        ax4 = fig.add_subplot(2, 2, 4)
        mid_i = self.nx * 3 // 4
        u_num = self.u[mid_i, :]
        u_ana = self.analytical_u()
        ax4.plot(u_ana, self.y, 'k--', lw=2, label='Analytical (Poiseuille)')
        ax4.plot(u_num, self.y, 'r-o', ms=4, lw=1.5, label=f'Numerical (x={self.x[mid_i]:.2f})')
        ax4.set_xlabel('u (m/s)')
        ax4.set_ylabel('y (m)')
        ax4.set_title(f'Velocity Profile at x = {self.x[mid_i]:.2f} m')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  图片已保存: {save_path}")

        # 额外: 中心线分布
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle("Centerline Profiles", fontsize=13, fontweight='bold')

        mid_j = self.ny // 2
        u_max_ana = np.max(u_ana)
        ax5 = axes2[0]
        ax5.plot(self.x, self.u[:, mid_j], 'b-', lw=2, label='u centerline (Numerical)')
        ax5.axhline(y=u_max_ana, color='k', ls='--', alpha=0.6,
                    label=f'u_max analytical = {u_max_ana:.4f}')
        ax5.set_xlabel('x (m)')
        ax5.set_ylabel('u (m/s)')
        ax5.set_title('Centerline u-velocity Development')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        ax6 = axes2[1]
        ax6.plot(self.x, self.p[:, mid_j], 'r-', lw=2, label='p centerline (Numerical)')
        # 充分发展段理论压降斜率 dp/dx = -12 * mu * u_mean / Ly^2
        dpdx_ana = -8.0 * self.mu * u_max_ana / (self.Ly ** 2)
        p_ana_line = dpdx_ana * (self.x - self.Lx)
        ax6.plot(self.x, p_ana_line, 'k--', lw=1.5, alpha=0.6,
                 label=f'Analytical Fully-Developed Slope ({dpdx_ana:.3f} Pa/m)')
        ax6.set_xlabel('x (m)')
        ax6.set_ylabel('p (Pa)')
        ax6.set_title('Centerline Pressure')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            save_path2 = save_path.replace('.png', '_centerline.png')
            fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
            print(f"  图片已保存: {save_path2}")

        plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    # ===== 可调参数 =====
    NX   = 101        # x 方向网格数
    NY   = 21         # y 方向网格数
    LX   = 10.0       # 槽道长度 (m)
    LY   = 1.0        # 槽道高度 (m)
    RHO  = 1.0        # 密度 (kg/m^3)
    MU   = 0.01       # 动力粘度 (Pa·s)  → Re = rho*u_in*Ly/mu = 100
    U_IN = 1.0        # 入口速度 (m/s)

    solver = ChannelFlowSIMPLE(
        nx=NX, ny=NY, Lx=LX, Ly=LY,
        rho=RHO, mu=MU, u_in=U_IN
    )

    solver.solve(
        max_iter=4000,
        tol=1e-6,
        alpha_u=0.6,      # 动量亚松弛
        alpha_v=0.6,
        alpha_p=0.2,      # 压力亚松弛
    )

    solver.plot_results(save_path='./channel_flow_result.png')