# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 2D 槽道流求解器 —— 泛化神经网络多网格测试与网格线可视化
# ================================================================
# 改进要点:
#   1. 支持对多个未知的测试网格进行连续评估。
#   2. 独立绘制每个测试网格的对比图。
#   3. 在云图中显式叠加物理网格线，直观展示当前计算网格的粗糙程度。
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from scipy.interpolate import RectBivariateSpline

# # ============================================================
# # 向量化 Thomas 算法
# # ============================================================
# def tdma_batch(low, diag, up, rhs):
#     N, M = rhs.shape
#     cp = np.zeros_like(rhs)
#     dp = np.zeros_like(rhs)
#     cp[:, 0] = up[:, 0] / diag[:, 0]
#     dp[:, 0] = rhs[:, 0] / diag[:, 0]
#     for k in range(1, M):
#         w = diag[:, k] - low[:, k] * cp[:, k - 1]
#         cp[:, k] = up[:, k] / w
#         dp[:, k] = (rhs[:, k] - low[:, k] * dp[:, k - 1]) / w
#     x = np.zeros_like(rhs)
#     x[:, -1] = dp[:, -1]
#     for k in range(M - 2, -1, -1):
#         x[:, k] = dp[:, k] - cp[:, k] * x[:, k + 1]
#     return x

# # ============================================================
# # 主求解器 (保持底层 SIMPLE 算法逻辑)
# # ============================================================
# class ChannelFlowSIMPLE:
#     def __init__(self, nx=51, ny=21, Lx=10.0, Ly=1.0, rho=1.0, mu=0.01, u_in=1.0):
#         self.nx, self.ny = nx, ny
#         self.Lx, self.Ly = Lx, Ly
#         self.dx = Lx / (nx - 1)
#         self.dy = Ly / (ny - 1)
#         self.x = np.linspace(0, Lx, nx)
#         self.y = np.linspace(0, Ly, ny)
#         self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
#         self.rho, self.mu, self.u_in = rho, mu, u_in

#         self.u = np.zeros((nx, ny))
#         self.v = np.zeros((nx, ny))
#         self.p = np.zeros((nx, ny))

#         sh = (nx, ny)
#         self.aE = np.zeros(sh); self.aW = np.zeros(sh)
#         self.aN = np.zeros(sh); self.aS = np.zeros(sh)
#         self.aP = np.zeros(sh); self.Sp = np.zeros(sh)

#         # 神经网络源项
#         self.Sp_nn_u = None
#         self.Sp_nn_v = None
#         self.Sp_nn_div = None

#         self._init_fields()

#     def _init_fields(self):
#         self.u[:, :] = self.u_in
#         self.v[:, :] = 0.0
#         self.p[:, :] = 0.0
#         self.u[:, 0] = self.u[:, -1] = 0.0
#         self.u[0, 0] = self.u[-1, 0] = self.u[0, -1] = self.u[-1, -1] = 0.0

#     def _bc_u(self):
#         self.u[0, 1:-1] = self.u_in
#         self.u[-1, 1:-1] = self.u[-2, 1:-1]
#         self.u[:, 0] = self.u[:, -1] = 0.0

#     def _bc_v(self):
#         self.v[0, :] = 0.0
#         self.v[-1, :] = self.v[-2, :]
#         self.v[:, 0] = self.v[:, -1] = 0.0

#     def _bc_p(self):
#         self.p[-1, :] = 0.0
#         self.p[0, :] = self.p[1, :]
#         self.p[:, 0] = self.p[:, 1]
#         self.p[:, -1] = self.p[:, -2]

#     def _bc_p_prime(self, pp):
#         pp[-1, :] = 0.0
#         pp[0, :] = pp[1, :]
#         pp[:, 0] = pp[:, 1]
#         pp[:, -1] = pp[:, -2]

#     @staticmethod
#     def _gradient(f, dx, dy):
#         Gx = np.zeros_like(f); Gy = np.zeros_like(f)
#         Gx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
#         Gy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
#         Gx[0, :]  = (-3 * f[0, :]  + 4 * f[1, :]  - f[2, :])  / (2 * dx)
#         Gx[-1, :] = ( 3 * f[-1, :] - 4 * f[-2, :] + f[-3, :]) / (2 * dx)
#         Gy[:, 0]  = (-3 * f[:, 0]  + 4 * f[:, 1]  - f[:, 2])  / (2 * dy)
#         Gy[:, -1] = ( 3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * dy)
#         return Gx, Gy

#     def _assemble_momentum(self, direction):
#         dx, dy, rho, mu = self.dx, self.dy, self.rho, self.mu
#         sl = (slice(1, -1), slice(1, -1))
#         ui = self.u[sl]; vi = self.v[sl]

#         De = Dw = mu / dx ** 2
#         Dn = Ds = mu / dy ** 2

#         aE_c = np.where(ui >= 0, 0.0, -rho * ui / dx)
#         aW_c = np.where(ui >= 0,  rho * ui / dx, 0.0)
#         aN_c = np.where(vi >= 0, 0.0, -rho * vi / dy)
#         aS_c = np.where(vi >= 0,  rho * vi / dy, 0.0)

#         self.aE[sl] = De + aE_c; self.aW[sl] = Dw + aW_c
#         self.aN[sl] = Dn + aN_c; self.aS[sl] = Ds + aS_c
#         self.aP[sl] = self.aE[sl] + self.aW[sl] + self.aN[sl] + self.aS[sl]

#         Gx, Gy = self._gradient(self.p, dx, dy)
#         if direction == 'u':
#             self.Sp[sl] = -Gx[sl]
#             if self.Sp_nn_u is not None: self.Sp[sl] += self.Sp_nn_u[sl]
#         else:
#             self.Sp[sl] = -Gy[sl]
#             if self.Sp_nn_v is not None: self.Sp[sl] += self.Sp_nn_v[sl]

#     def _solve_scalar(self, phi, alpha):
#         sl = (slice(1, -1), slice(1, -1))
#         phi_old = phi.copy()
#         aP_ur = self.aP.copy(); S_ur  = self.Sp.copy()
#         aP_ur[sl] = self.aP[sl] / alpha
#         S_ur[sl]  = self.Sp[sl] + (1.0 - alpha) / alpha * self.aP[sl] * phi_old[sl]

#         # x 扫掠
#         low  = -self.aW[sl].T.copy(); diag = aP_ur[sl].T.copy(); up   = -self.aE[sl].T.copy()
#         rhs  = (self.aN[sl] * phi[1:-1, 2:] + self.aS[sl] * phi[1:-1, :-2] + S_ur[sl]).T
#         rhs[:, 0]  += self.aW[1, 1:-1] * phi[0, 1:-1]; diag[:, -1] -= self.aE[-2, 1:-1]
#         phi[1:-1, 1:-1] = tdma_batch(low, diag, up, rhs).T
#         phi[-1, 1:-1] = phi[-2, 1:-1]

#         # y 扫掠
#         low  = -self.aS[sl].copy(); diag = aP_ur[sl].copy(); up   = -self.aN[sl].copy()
#         rhs  = (self.aE[sl] * phi[2:, 1:-1] + self.aW[sl] * phi[:-2, 1:-1] + S_ur[sl])
#         rhs[:, 0]  += self.aS[1:-1, 1] * phi[1:-1, 0]; rhs[:, -1] += self.aN[1:-1, -2] * phi[1:-1, -1]
#         phi[1:-1, 1:-1] = tdma_batch(low, diag, up, rhs)

#     def _rc_face_velocities(self, alpha_u):
#         nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
#         d = np.zeros((nx, ny))
#         d[1:-1, 1:-1] = alpha_u / self.aP[1:-1, 1:-1]
#         d[0, :] = d[1, :]; d[-1, :] = d[-2, :]; d[:, 0] = d[:, 1]; d[:, -1] = d[:, -2]

#         Gx, Gy = self._gradient(self.p, dx, dy)
#         d_bar_e = 0.5 * (d[:-1, :] + d[1:, :])
#         Gx_bar_e = 0.5 * (Gx[:-1, :] + Gx[1:, :])
#         Gx_compact_e = (self.p[1:, :] - self.p[:-1, :]) / dx
#         u_e = 0.5 * (self.u[:-1, :] + self.u[1:, :]) + d_bar_e * (Gx_bar_e - Gx_compact_e)

#         d_bar_n = 0.5 * (d[:, :-1] + d[:, 1:])
#         Gy_bar_n = 0.5 * (Gy[:, :-1] + Gy[:, 1:])
#         Gy_compact_n = (self.p[:, 1:] - self.p[:, :-1]) / dy
#         v_n = 0.5 * (self.v[:, :-1] + self.v[:, 1:]) + d_bar_n * (Gy_bar_n - Gy_compact_n)

#         return u_e, v_n, d

#     def _solve_pressure_correction(self, alpha_u, n_pc_sweeps=25):
#         nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
#         u_e, v_n, d = self._rc_face_velocities(alpha_u)

#         div = np.zeros((nx, ny))
#         sl = (slice(1, nx - 1), slice(1, ny - 1))
#         div[sl] = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / dx + (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / dy

#         b = -(div[sl] - self.Sp_nn_div[sl]) if self.Sp_nn_div is not None else -div[sl]

#         aE_pc = 0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / dx ** 2
#         aW_pc = 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / dx ** 2
#         aN_pc = 0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / dy ** 2
#         aS_pc = 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / dy ** 2
#         aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

#         aP_pc[0, :]  -= aW_pc[0, :];  aW_pc[0, :]  = 0.0
#         aP_pc[:, 0]  -= aS_pc[:, 0];  aS_pc[:, 0]  = 0.0
#         aP_pc[:, -1] -= aN_pc[:, -1]; aN_pc[:, -1] = 0.0
#         aE_pc[-1, :] = 0.0

#         pp = np.zeros((nx, ny))
#         low_x, diag_x, up_x = -aW_pc.T, aP_pc.T, -aE_pc.T
#         low_y, diag_y, up_y = -aS_pc,   aP_pc,   -aN_pc

#         for _ in range(n_pc_sweeps):
#             rhs_x = (aN_pc * pp[1:-1, 2:] + aS_pc * pp[1:-1, :-2] + b).T
#             pp[1:-1, 1:-1] = tdma_batch(low_x, diag_x, up_x, rhs_x).T
#             rhs_y = (aE_pc * pp[2:, 1:-1] + aW_pc * pp[:-2, 1:-1] + b)
#             pp[1:-1, 1:-1] = tdma_batch(low_y, diag_y, up_y, rhs_y)

#         self._bc_p_prime(pp)
#         return pp, d, np.max(np.abs(b))

#     def _correct_fields(self, pp, d, alpha_p):
#         dx, dy = self.dx, self.dy
#         sl = (slice(1, -1), slice(1, -1))
#         self.p[sl] += alpha_p * pp[sl]
#         Gx_pp, Gy_pp = self._gradient(pp, dx, dy)
#         self.u[sl] -= d[sl] * Gx_pp[sl]; self.v[sl] -= d[sl] * Gy_pp[sl]
#         self._bc_u(); self._bc_v(); self._bc_p()

#     def solve(self, max_iter=4000, tol=1e-6, alpha_u=0.6, alpha_v=0.6, alpha_p=0.2, verbose=True):
#         if verbose:
#             print(f"[{self.nx}x{self.ny}] 求解中...", end=" ")
#         for it in range(1, max_iter + 1):
#             self._assemble_momentum('u'); self._solve_scalar(self.u, alpha_u); self._bc_u()
#             self._assemble_momentum('v'); self._solve_scalar(self.v, alpha_v); self._bc_v()
#             pp, d, mass_resid = self._solve_pressure_correction(alpha_u)
#             self._correct_fields(pp, d, alpha_p)
#             if mass_resid < tol and it > 15:
#                 if verbose: print(f"收敛于 {it} 步 (Err={mass_resid:.2e})")
#                 break
#         else:
#             if verbose: print(f"达到最大迭代次数 (Err={mass_resid:.2e})")
#         return mass_resid

# # ============================================================
# # 完备离散残差计算器
# # ============================================================
# def compute_discrete_residual(solver, u_s, v_s, p_s, alpha_u=0.6):
#     solver.u, solver.v, solver.p = u_s.copy(), v_s.copy(), p_s.copy()
#     solver._bc_u(); solver._bc_v(); solver._bc_p()
#     sl = (slice(1, -1), slice(1, -1))

#     solver._assemble_momentum('u')
#     Gx, Gy = solver._gradient(solver.p, solver.dx, solver.dy)
#     Sp_u, Sp_v = -Gx[sl], -Gy[sl]

#     u_eval, v_eval = solver.u.copy(), solver.v.copy()
#     u_eval[-1, 1:-1] = u_eval[-2, 1:-1]; v_eval[-1, 1:-1] = v_eval[-2, 1:-1]

#     Res_u = (solver.aP[sl]*u_eval[sl] - solver.aE[sl]*u_eval[2:,1:-1] - solver.aW[sl]*u_eval[:-2,1:-1] 
#              - solver.aN[sl]*u_eval[1:-1,2:] - solver.aS[sl]*u_eval[1:-1,:-2] - Sp_u)
#     Res_v = (solver.aP[sl]*v_eval[sl] - solver.aE[sl]*v_eval[2:,1:-1] - solver.aW[sl]*v_eval[:-2,1:-1] 
#              - solver.aN[sl]*v_eval[1:-1,2:] - solver.aS[sl]*v_eval[1:-1,:-2] - Sp_v)

#     u_e, v_n, _ = solver._rc_face_velocities(alpha_u)
#     Res_div = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / solver.dx + (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / solver.dy

#     return Res_u, Res_v, Res_div

# # ============================================================
# # 通用化泛化神经网络补偿器 (支持跨粗糙度预测)
# # ============================================================
# class GeneralizedNeuralCorrector:
#     def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         self.device = device
#         self.models = {}
#         self.stats = {} 

#     def build_model(self, input_dim=4, hidden_layers=[128, 128, 64]):
#         layers = []
#         prev_dim = input_dim
#         for h in hidden_layers:
#             layers.append(nn.Linear(prev_dim, h))
#             layers.append(nn.Tanh())
#             prev_dim = h
#         layers.append(nn.Linear(prev_dim, 1))
#         return nn.Sequential(*layers).to(self.device)

#     def prepare_multi_grid_data(self, datasets):
#         X_all = np.vstack([d['X'] for d in datasets])
#         X_train = torch.tensor(X_all, dtype=torch.float32, device=self.device)

#         targets = {
#             'u': np.concatenate([d['u'] for d in datasets]),
#             'v': np.concatenate([d['v'] for d in datasets]),
#             'div': np.concatenate([d['div'] for d in datasets])
#         }

#         Y_tensors = {}
#         for k, val in targets.items():
#             mean, std = float(np.mean(val)), float(np.std(val)) + 1e-12
#             self.stats[k] = (mean, std)
#             norm_val = (val - mean) / std
#             Y_tensors[k] = torch.tensor(norm_val, dtype=torch.float32, device=self.device).unsqueeze(1)
            
#         return X_train, Y_tensors

#     def train(self, X_train, Y_tensors, epochs=3000, lr=2e-3):
#         print(f"\n[{'='*50}]")
#         print(f"开始训练多尺度泛化残差预测网络 (总样本点: {len(X_train)})")
#         print(f"[{'='*50}]")

#         criterion = nn.MSELoss()
#         optimizers = {}; schedulers = {}

#         for k in ['u', 'v', 'div']:
#             self.models[k] = self.build_model(input_dim=4, hidden_layers=[128, 128, 64])
#             optimizers[k] = torch.optim.Adam(self.models[k].parameters(), lr=lr, weight_decay=1e-6)
#             schedulers[k] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[k], T_max=epochs, eta_min=1e-5)

#         for epoch in range(1, epochs + 1):
#             losses = {}
#             for k in ['u', 'v', 'div']:
#                 optimizers[k].zero_grad()
#                 pred = self.models[k](X_train)
#                 loss = criterion(pred, Y_tensors[k])
#                 loss.backward()
#                 optimizers[k].step()
#                 schedulers[k].step()
#                 losses[k] = loss.item()

#             if epoch % 500 == 0 or epoch == 1:
#                 cur_lr = optimizers['u'].param_groups[0]['lr']
#                 print(f" Epoch {epoch:4d} | MSE_u={losses['u']:.3e} | MSE_v={losses['v']:.3e} | "
#                       f"MSE_div={losses['div']:.3e} | LR={cur_lr:.2e}")

#     def predict(self, solver):
#         nx, ny = solver.nx, solver.ny
#         X_norm = solver.X.ravel() / solver.Lx
#         Y_norm = solver.Y.ravel() / solver.Ly
#         dx_norm = solver.dx / solver.Lx
#         dy_norm = solver.dy / solver.Ly

#         X_all = np.column_stack([
#             X_norm, Y_norm,
#             np.full_like(X_norm, dx_norm),
#             np.full_like(X_norm, dy_norm)
#         ])
#         X_tensor = torch.tensor(X_all, dtype=torch.float32, device=self.device)

#         preds = {}
#         with torch.no_grad():
#             for k in ['u', 'v', 'div']:
#                 mean, std = self.stats[k]
#                 raw = self.models[k](X_tensor).cpu().numpy().reshape(nx, ny)
#                 field = raw * std + mean
#                 field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0
#                 preds[k] = field

#         return preds['u'], preds['v'], preds['div']


# # ============================================================
# # 独立绘图函数 (包含网格线可视化)
# # ============================================================
# def plot_generalization_results(solver_dense, solver_test_orig, solver_test_corr, u_ref, test_grid):
#     fig, axes = plt.subplots(2, 2, figsize=(16, 10))
#     fig.suptitle(f"Zero-Shot Generalization Test on Unseen Grid: {test_grid}", fontsize=15, fontweight='bold')
    
#     # 获取速度剖面与误差评估截面索引 (x ≈ 7.5)
#     i_orig = np.argmin(np.abs(solver_test_orig.x - 7.5))
#     i_dense = np.argmin(np.abs(solver_dense.x - 7.5))
#     x_pos = solver_test_orig.x[i_orig]

#     # --- 1. 原始粗网格云图 (带网格线) ---
#     ax = axes[0, 0]
#     cf1 = ax.contourf(solver_test_orig.X.T, solver_test_orig.Y.T, solver_test_orig.u.T, levels=30, cmap='viridis')
#     plt.colorbar(cf1, ax=ax, label='u (m/s)')
#     # 叠加水平和垂直物理网格线
#     for x_val in solver_test_orig.x:
#         ax.axvline(x_val, color='k', lw=0.4, alpha=0.4)
#     for y_val in solver_test_orig.y:
#         ax.axhline(y_val, color='k', lw=0.4, alpha=0.4)
#     ax.set_title(f'Original Coarse ({test_grid[0]}x{test_grid[1]}) + Physical Mesh')
#     ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
#     ax.set_aspect('equal')

#     # --- 2. 神经网络修正云图 (带网格线) ---
#     ax = axes[0, 1]
#     cf2 = ax.contourf(solver_test_corr.X.T, solver_test_corr.Y.T, solver_test_corr.u.T, 
#                       levels=30, cmap='viridis', vmin=solver_dense.u.min(), vmax=solver_dense.u.max())
#     plt.colorbar(cf2, ax=ax, label='u (m/s)')
#     # 叠加水平和垂直物理网格线
#     for x_val in solver_test_corr.x:
#         ax.axvline(x_val, color='k', lw=0.4, alpha=0.4)
#     for y_val in solver_test_corr.y:
#         ax.axhline(y_val, color='k', lw=0.4, alpha=0.4)
#     ax.set_title(f'NN-Corrected Coarse ({test_grid[0]}x{test_grid[1]}) + Physical Mesh')
#     ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
#     ax.set_aspect('equal')

#     # --- 3. 截面速度剖面 ---
#     ax = axes[1, 0]
#     ax.plot(solver_dense.u[i_dense, :], solver_dense.y, 'k-', lw=2.5, label='Dense Reference')
#     ax.plot(solver_test_orig.u[i_orig, :], solver_test_orig.y, 'r--o', ms=5, label='Original Coarse')
#     ax.plot(solver_test_corr.u[i_orig, :], solver_test_corr.y, 'b:s', ms=5, label='NN-Corrected Coarse')
#     ax.set_title(f"Velocity Profile at x = {x_pos:.2f} m")
#     ax.set_xlabel('u (m/s)'); ax.set_ylabel('y (m)')
#     ax.legend(); ax.grid(True, alpha=0.3)

#     # --- 4. 绝对误差对数分布 ---
#     ax = axes[1, 1]
#     abs_err_orig = np.abs(u_ref[i_orig, :] - solver_test_orig.u[i_orig, :])
#     abs_err_corr = np.abs(u_ref[i_orig, :] - solver_test_corr.u[i_orig, :])
#     ax.semilogy(solver_test_orig.y, abs_err_orig + 1e-12, 'r-o', lw=2, label='Original Error')
#     ax.semilogy(solver_test_orig.y, abs_err_corr + 1e-12, 'b-s', lw=2, label='NN-Corrected Error')
#     ax.set_title(f"Pointwise Absolute Error at x = {x_pos:.2f} m")
#     ax.set_xlabel('y (m)'); ax.set_ylabel('|Error| (m/s)')
#     ax.legend(); ax.grid(True, alpha=0.3, which='both')

#     plt.tight_layout()
#     plt.savefig(f'./Generalization_Test_{test_grid[0]}x{test_grid[1]}.png', dpi=150)
#     plt.show()


# # ============================================================
# # 主测试流程 (多组网格测试)
# # ============================================================
# if __name__ == '__main__':
#     LX, LY = 10.0, 1.0
#     RHO, MU, U_IN = 1.0, 0.01, 1.0
#     ALPHA_U, ALPHA_V, ALPHA_P = 0.6, 0.6, 0.2
    
#     # 密网格基准
#     DENSE_GRID = (81, 41)
#     # 用于提取残差训练集的多尺度网格
#     TRAIN_GRIDS = [(15, 8), (21, 11), (31, 16), (41, 21)]  
    
#     # ✅ 改进点：提供 3 组全新的测试网格！分别代表 极粗、中粗、较密
#     TEST_GRIDS = [
#         (19, 10), # 极粗 (在 15x8 和 21x11 之间)
#         (25, 13), # 中粗 (在 21x11 和 31x16 之间)
#         (35, 18)  # 较密 (在 31x16 和 41x21 之间)
#     ]

#     print(f"\n>>> Step 1: 密网格基准求解 {DENSE_GRID} <<<")
#     solver_dense = ChannelFlowSIMPLE(*DENSE_GRID, LX, LY, RHO, MU, U_IN)
#     solver_dense.solve(max_iter=4000)
#     interp_u = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.u)
#     interp_v = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.v)
#     interp_p = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.p)

#     print(f"\n>>> Step 2: 收集多尺度粗网格特征与残差 <<<")
#     datasets = []
#     for (nx, ny) in TRAIN_GRIDS:
#         print(f" -> 正在处理训练网格: {nx} x {ny}")
#         solver_c = ChannelFlowSIMPLE(nx, ny, LX, LY, RHO, MU, U_IN)
#         u_s = interp_u(solver_c.x, solver_c.y)
#         v_s = interp_v(solver_c.x, solver_c.y)
#         p_s = interp_p(solver_c.x, solver_c.y)
#         res_u, res_v, res_div = compute_discrete_residual(solver_c, u_s, v_s, p_s, alpha_u=ALPHA_U)
        
#         # 提取当前网格的 4 维特征 (内部节点)
#         X_norm = solver_c.X[1:-1, 1:-1].ravel() / solver_c.Lx
#         Y_norm = solver_c.Y[1:-1, 1:-1].ravel() / solver_c.Ly
#         dx_norm, dy_norm = solver_c.dx / solver_c.Lx, solver_c.dy / solver_c.Ly
        
#         X_features = np.column_stack([
#             X_norm, Y_norm, 
#             np.full_like(X_norm, dx_norm), 
#             np.full_like(Y_norm, dy_norm)
#         ])
        
#         datasets.append({
#             'X': X_features,
#             'u': res_u.ravel(), 'v': res_v.ravel(), 'div': res_div.ravel()
#         })

#     print(f"\n>>> Step 3: 训练通用多尺度神经网络 <<<")
#     corrector = GeneralizedNeuralCorrector()
#     X_train, Y_tensors = corrector.prepare_multi_grid_data(datasets)
#     corrector.train(X_train, Y_tensors, epochs=3000, lr=2e-3)

#     print("\n" + "="*60)
#     print(">>> Step 4: 遍历并测试多组【未知】的粗糙网格 <<<")
#     print("="*60)

#     # 循环对 3 组新网格进行泛化测试并绘图
#     for test_grid in TEST_GRIDS:
#         print(f"\n>>> 开始测试未见过的网格: {test_grid} <<<")
        
#         # 1. 原始粗网格求解
#         solver_test_orig = ChannelFlowSIMPLE(*test_grid, LX, LY, RHO, MU, U_IN)
#         solver_test_orig.solve(verbose=False)
        
#         # 2. 神经网络修正粗网格求解
#         solver_test_corr = ChannelFlowSIMPLE(*test_grid, LX, LY, RHO, MU, U_IN)
#         res_u_pred, res_v_pred, res_div_pred = corrector.predict(solver_test_corr)
#         solver_test_corr.Sp_nn_u = res_u_pred
#         solver_test_corr.Sp_nn_v = res_v_pred
#         solver_test_corr.Sp_nn_div = res_div_pred
#         solver_test_corr.solve(verbose=False)

#         # 误差评估
#         u_ref = interp_u(solver_test_orig.x, solver_test_orig.y)
#         err_orig = np.max(np.abs(u_ref[1:-1, 1:-1] - solver_test_orig.u[1:-1, 1:-1]))
#         err_corr = np.max(np.abs(u_ref[1:-1, 1:-1] - solver_test_corr.u[1:-1, 1:-1]))
        
#         print(f"  [评估] {test_grid} 原始误差: {err_orig:.6f}")
#         print(f"  [评估] {test_grid} NN修正后 : {err_corr:.6f} (降低 {(1.0 - err_corr/err_orig)*100:.2f}%)")

#         # 独立为每一组调用绘图函数
#         plot_generalization_results(solver_dense, solver_test_orig, solver_test_corr, u_ref, test_grid)
#         print("-" * 60)
        
#

# GPT
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 槽道流求解器 —— 多粗糙度网格残差学习与神经网络误差补偿
================================================================
修复要点:
  1. 修复底层 SIMPLE 求解器的 3 个 Bug (TDMA 边界移项、压力修正边界系数与内迭代).
  2. 引入【动量 + 连续性(散度)】双残差补偿机制 (Dual-Consistency):
     不仅补偿粗网格动量方程截断误差 (Res_u, Res_v),
     同时补偿粗网格 Rhie-Chow 连续性方程截断误差 (Res_div), 防止压力修正冲毁动量修正.
  3. 保持采样场压力边界与粗网格算子自洽.
  4. 多种粗网格统一采样密网格参考解，计算动量与 Rhie-Chow 连续性离散残差。
  5. 网络输入包含坐标与无量纲网格粗糙度；多输出联合训练、网格均衡加权、留一分辨率验证。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
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
# 多粗糙度神经网络误差补偿器
# ============================================================
class NeuralCorrector:
    """单一多输出网络：输入 (x/Lx, y/Ly, log2(dx/Lx), log2(dy/Ly))。

    三个输出分别对应动量残差 Res_u、Res_v 和 Rhie-Chow 连续性残差 Res_div。
    训练时每一种网格的总权重相同，避免节点数较多的网格支配损失函数。
    """

    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    @staticmethod
    def _raw_features(solver, interior_only=True):
        if interior_only:
            X = solver.X[1:-1, 1:-1]
            Y = solver.Y[1:-1, 1:-1]
        else:
            X = solver.X
            Y = solver.Y

        # 坐标无量纲化；用网格间距的 log2 特征表示粗糙程度。
        # 该定义不依赖训练时的某个固定密网格，因而可直接用于新网格。
        hx_feature = np.log2(max(solver.dx / solver.Lx, 1e-12))
        hy_feature = np.log2(max(solver.dy / solver.Ly, 1e-12))
        return np.column_stack([
            (X / solver.Lx).ravel(),
            (Y / solver.Ly).ravel(),
            np.full(X.size, hx_feature),
            np.full(X.size, hy_feature),
        ])

    @staticmethod
    def _weighted_mean_std(values, weights):
        weights = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        values = np.asarray(values, dtype=np.float64)
        denom = max(float(weights.sum()), 1e-12)
        mean = (values * weights).sum(axis=0) / denom
        var = (((values - mean) ** 2) * weights).sum(axis=0) / denom
        std = np.sqrt(np.maximum(var, 0.0))
        std[std < 1e-12] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def _collect_arrays(cases):
        """cases: {grid_key: {'solver': solver, 'residuals': (res_u,res_v,res_div)}}"""
        X_parts, Y_parts, W_parts = [], [], []
        for grid_key, case in cases.items():
            solver = case['solver']
            res_u, res_v, res_div = case['residuals']
            Xg = NeuralCorrector._raw_features(solver, interior_only=True)
            Yg = np.column_stack([res_u.ravel(), res_v.ravel(), res_div.ravel()])
            if len(Xg) != len(Yg):
                raise ValueError(
                    f'网格 {grid_key} 的特征数 {len(Xg)} 与残差数 {len(Yg)} 不一致。'
                )
            # 每个网格内部的权重和为 1，因此每个网格对训练目标贡献相等。
            Wg = np.full((len(Xg),), 1.0 / max(len(Xg), 1), dtype=np.float32)
            X_parts.append(Xg.astype(np.float32))
            Y_parts.append(Yg.astype(np.float32))
            W_parts.append(Wg)

        if not X_parts:
            raise ValueError('训练数据为空，请检查 TRAIN_GRIDS 配置。')
        return np.concatenate(X_parts), np.concatenate(Y_parts), np.concatenate(W_parts)

    def prepare_datasets(self, train_cases, validation_cases=None):
        """仅用训练网格拟合标准化参数，再变换训练集与独立验证网格。"""
        X_train_raw, Y_train_raw, W_train = self._collect_arrays(train_cases)
        self.x_mean, self.x_std = self._weighted_mean_std(X_train_raw, W_train)
        self.y_mean, self.y_std = self._weighted_mean_std(Y_train_raw, W_train)

        def to_tensors(X_raw, Y_raw, weights):
            Xn = (X_raw - self.x_mean) / self.x_std
            Yn = (Y_raw - self.y_mean) / self.y_std
            return (
                torch.as_tensor(Xn, dtype=torch.float32, device=self.device),
                torch.as_tensor(Yn, dtype=torch.float32, device=self.device),
                torch.as_tensor(weights, dtype=torch.float32, device=self.device),
            )

        train_data = to_tensors(X_train_raw, Y_train_raw, W_train)
        val_data = None
        if validation_cases:
            X_val_raw, Y_val_raw, W_val = self._collect_arrays(validation_cases)
            val_data = to_tensors(X_val_raw, Y_val_raw, W_val)
        return train_data, val_data

    @staticmethod
    def build_model(input_dim=4, output_dim=3, hidden_layers=(96, 96, 96)):
        layers = []
        prev_dim = input_dim
        for width in hidden_layers:
            layers.extend([nn.Linear(prev_dim, width), nn.Tanh()])
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _weighted_loss(pred, target, weights):
        # 先对每个样本的三个标准化输出求平均，再按网格均衡权重求和。
        per_sample = ((pred - target) ** 2).mean(dim=1)
        return (per_sample * weights).sum() / torch.clamp(weights.sum(), min=1e-12)

    def train(self, train_data, validation_data=None, epochs=3000, lr=1.5e-3):
        X_train, Y_train, W_train = train_data
        self.model = self.build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=1e-5
        )

        print('\n' + '=' * 68)
        print('多粗糙度联合训练 | 输入: x/Lx, y/Ly, log2(dx/Lx), log2(dy/Ly)')
        print(f'训练样本数: {len(X_train)} | 使用网格均衡权重（每个网格总权重相同）')
        print('=' * 68)

        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = self.model(X_train)
            loss = self._weighted_loss(prediction, Y_train, W_train)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch == 1 or epoch % 250 == 0 or epoch == epochs:
                self.model.eval()
                with torch.no_grad():
                    train_loss = self._weighted_loss(self.model(X_train), Y_train, W_train).item()
                    if validation_data is not None:
                        X_val, Y_val, W_val = validation_data
                        val_loss = self._weighted_loss(self.model(X_val), Y_val, W_val).item()
                        val_text = f' | held-out MSE={val_loss:.4e}'
                    else:
                        val_text = ''
                cur_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch:4d} | train MSE={train_loss:.4e}'
                      f'{val_text} | LR={cur_lr:.2e}')

        print('✓ 多网格残差模型训练完成。')

    def predict(self, solver):
        if self.model is None or any(v is None for v in
                                     (self.x_mean, self.x_std, self.y_mean, self.y_std)):
            raise RuntimeError('模型尚未训练或加载，不能预测。')

        nx, ny = solver.nx, solver.ny
        X_all = self._raw_features(solver, interior_only=False)
        Xn = (X_all - self.x_mean) / self.x_std
        X_tensor = torch.as_tensor(Xn, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(X_tensor).cpu().numpy()
        pred = pred_norm * self.y_std + self.y_mean
        pred = pred.reshape(nx, ny, 3)

        # 边界源项清零；与原 SIMPLE 求解器的内部控制体设置保持一致。
        fields = [pred[:, :, i].copy() for i in range(3)]
        for field in fields:
            field[0, :] = field[-1, :] = 0.0
            field[:, 0] = field[:, -1] = 0.0
        return fields[0], fields[1], fields[2]

    def save(self, path):
        """保存模型与标准化参数，之后可在不同分辨率网格上复用。"""
        if self.model is None:
            raise RuntimeError('模型尚未训练，不能保存。')
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'x_mean': torch.as_tensor(self.x_mean),
            'x_std': torch.as_tensor(self.x_std),
            'y_mean': torch.as_tensor(self.y_mean),
            'y_std': torch.as_tensor(self.y_std),
        }
        torch.save(checkpoint, path)
        print(f'模型与标准化参数已保存: {path}')

    def load(self, path):
        """加载已保存模型；输入特征定义必须与本类一致。"""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # 兼容不支持 weights_only 的较旧 PyTorch
            checkpoint = torch.load(path, map_location=self.device)
        self.model = self.build_model().to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.x_mean = checkpoint['x_mean'].cpu().numpy().astype(np.float32)
        self.x_std = checkpoint['x_std'].cpu().numpy().astype(np.float32)
        self.y_mean = checkpoint['y_mean'].cpu().numpy().astype(np.float32)
        self.y_std = checkpoint['y_std'].cpu().numpy().astype(np.float32)
        self.model.eval()
        print(f'已加载模型: {path}')




# ============================================================
# 可视化函数：每种测试网格单独输出图片；云图叠加真实网格线
# ============================================================
def draw_mesh_lines(ax, solver, color='k', alpha=0.30, linewidth=0.45, zorder=5):
    """在云图上叠加求解器的实际结构网格线。"""
    for yj in solver.y:
        ax.plot(solver.x, np.full_like(solver.x, yj), color=color,
                alpha=alpha, linewidth=linewidth, zorder=zorder)
    for xi in solver.x:
        ax.plot(np.full_like(solver.y, xi), solver.y, color=color,
                alpha=alpha, linewidth=linewidth, zorder=zorder)


def _finish_map_axis(ax, solver, title):
    ax.set_title(title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(solver.x[0], solver.x[-1])
    ax.set_ylim(solver.y[0], solver.y[-1])
    ax.set_aspect('equal', adjustable='box')


def plot_residual_comparison(solver_coarse, res_u_true, res_u_pred, output_dir, grid_label):
    """为单个网格单独绘制真实/预测/误差残差云图，并画出该网格线。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nx, ny = solver_coarse.nx, solver_coarse.ny
    res_true_full = np.zeros((nx, ny), dtype=float)
    res_true_full[1:-1, 1:-1] = res_u_true
    res_pred_full = np.asarray(res_u_pred).copy()
    res_error_full = res_pred_full - res_true_full

    # 使用同一色阶范围，有利于直接比较真实与预测残差。
    vmax = max(float(np.max(np.abs(res_true_full))),
               float(np.max(np.abs(res_pred_full))), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    fig.suptitle(f'Discrete momentum residual Res_u | grid {grid_label}',
                 fontsize=14, fontweight='bold')

    panels = [
        (res_true_full, 'True Res_u', 'RdBu_r', -vmax, vmax),
        (res_pred_full, 'NN predicted Res_u', 'RdBu_r', -vmax, vmax),
        (res_error_full, f'Prediction error | max={np.max(np.abs(res_error_full)):.2e}',
         'RdBu_r', -vmax, vmax),
    ]
    for ax, (data, title, cmap, vmin, vmax_local) in zip(axes, panels):
        cf = ax.contourf(solver_coarse.X.T, solver_coarse.Y.T, data.T,
                         levels=25, cmap=cmap, vmin=vmin, vmax=vmax_local)
        draw_mesh_lines(ax, solver_coarse)
        fig.colorbar(cf, ax=ax, shrink=0.82)
        _finish_map_axis(ax, solver_coarse, title)

    path = output_dir / f'residual_u_grid_{grid_label}.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存残差云图: {path}')


def plot_flow_comparisons(solver_dense, solver_coarse_orig, solver_coarse_corrected,
                          u_ref_on_coarse, output_dir, grid_label):
    """每种测试网格分别保存流场云图比较；每张云图都显示对应网格线。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    u_ref_on_coarse = np.asarray(u_ref_on_coarse)
    err_orig = np.abs(u_ref_on_coarse - solver_coarse_orig.u)
    err_corr = np.abs(u_ref_on_coarse - solver_coarse_corrected.u)
    u_min = min(float(solver_dense.u.min()), float(u_ref_on_coarse.min()),
                float(solver_coarse_orig.u.min()), float(solver_coarse_corrected.u.min()))
    u_max = max(float(solver_dense.u.max()), float(u_ref_on_coarse.max()),
                float(solver_coarse_orig.u.max()), float(solver_coarse_corrected.u.max()))
    err_max = max(float(err_orig.max()), float(err_corr.max()), 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(19, 8.5), constrained_layout=True)
    fig.suptitle(f'Channel-flow field comparison | tested grid {grid_label}',
                 fontsize=15, fontweight='bold')

    panels = [
        (axes[0, 0], solver_dense, solver_dense.u,
         f'Dense reference ({solver_dense.nx}x{solver_dense.ny})', 'viridis', u_min, u_max),
        (axes[0, 1], solver_coarse_orig, u_ref_on_coarse,
         'Dense reference sampled on coarse grid', 'viridis', u_min, u_max),
        (axes[0, 2], solver_coarse_orig, solver_coarse_orig.u,
         'Coarse SIMPLE baseline', 'viridis', u_min, u_max),
        (axes[1, 0], solver_coarse_corrected, solver_coarse_corrected.u,
         'NN-corrected coarse solution', 'viridis', u_min, u_max),
        (axes[1, 1], solver_coarse_orig, err_orig,
         f'Baseline absolute error | max={err_orig.max():.2e}', 'magma', 0.0, err_max),
        (axes[1, 2], solver_coarse_corrected, err_corr,
         f'Corrected absolute error | max={err_corr.max():.2e}', 'magma', 0.0, err_max),
    ]
    for ax, grid_solver, data, title, cmap, vmin, vmax in panels:
        cf = ax.contourf(grid_solver.X.T, grid_solver.Y.T, data.T,
                         levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
        draw_mesh_lines(ax, grid_solver,
                        linewidth=0.30 if grid_solver.nx > 60 else 0.50,
                        alpha=0.28 if grid_solver.nx > 60 else 0.42)
        fig.colorbar(cf, ax=ax, shrink=0.82)
        _finish_map_axis(ax, grid_solver, title)

    path = output_dir / f'flow_fields_grid_{grid_label}.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存带网格线的流场云图: {path}')


def plot_velocity_profiles(solver_dense, solver_coarse_orig, solver_coarse_corrected,
                           u_ref_on_coarse, output_dir, grid_label, x_profile=7.5):
    """为每个测试分辨率单独保存速度剖面及剖面误差图。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    i_dense = int(np.argmin(np.abs(solver_dense.x - x_profile)))
    i_coarse = int(np.argmin(np.abs(solver_coarse_orig.x - x_profile)))
    err_orig_profile = np.abs(u_ref_on_coarse[i_coarse, :] -
                              solver_coarse_orig.u[i_coarse, :])
    err_corr_profile = np.abs(u_ref_on_coarse[i_coarse, :] -
                              solver_coarse_corrected.u[i_coarse, :])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    fig.suptitle(f'Velocity profile at x ≈ {solver_coarse_orig.x[i_coarse]:.3f} m | grid {grid_label}',
                 fontsize=13, fontweight='bold')
    ax = axes[0]
    ax.plot(solver_dense.u[i_dense, :], solver_dense.y, 'k-', lw=2.4,
            label='Dense reference')
    ax.plot(solver_coarse_orig.u[i_coarse, :], solver_coarse_orig.y,
            'r--o', ms=4, lw=1.6, label='Coarse baseline')
    ax.plot(solver_coarse_corrected.u[i_coarse, :], solver_coarse_corrected.y,
            'b:s', ms=4, lw=1.8, label='NN corrected')
    ax.set_xlabel('u (m/s)'); ax.set_ylabel('y (m)')
    ax.set_title('Velocity profile')
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.semilogy(solver_coarse_orig.y, err_orig_profile + 1e-14, 'r-o',
                ms=4, lw=1.6, label='Baseline error')
    ax.semilogy(solver_coarse_orig.y, err_corr_profile + 1e-14, 'b-s',
                ms=4, lw=1.8, label='NN-corrected error')
    ax.set_xlabel('y (m)'); ax.set_ylabel('|u - u_ref| (m/s)')
    ax.set_title('Pointwise profile error')
    ax.grid(True, alpha=0.3, which='both'); ax.legend()

    path = output_dir / f'velocity_profile_grid_{grid_label}.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存速度剖面对比图: {path}')


# ============================================================
# 主程序：多种粗网格训练 + 一个验证网格 + 两个独立测试网格
# ============================================================
if __name__ == '__main__':
    from pathlib import Path

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    NX_DENSE, NY_DENSE = 81, 41
    TRAIN_GRIDS = [
        (11, 7),
        (15, 9),
        (21, 11),
        (41, 21),
        (51, 27),
    ]
    # 现有的留出验证分辨率保留；再新增两组只用于训练后测试的分辨率。
    VALIDATION_GRID = (31, 17)
    TEST_GRIDS = [
        (25, 13),
        (35, 19),
    ]
    EVAL_GRIDS = [VALIDATION_GRID] + TEST_GRIDS

    LX, LY = 10.0, 1.0
    RHO, MU, U_IN = 1.0, 0.01, 1.0
    ALPHA_U, ALPHA_V, ALPHA_P = 0.6, 0.6, 0.2
    MAX_ITER = 4000
    SOLVER_TOL = 1e-6
    TRAIN_EPOCHS = 3000
    MODEL_PATH = './channel_flow_multigrid_residual_nn.pt'
    OUTPUT_DIR = Path('./multigrid_test_results')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if VALIDATION_GRID in TRAIN_GRIDS:
        raise ValueError('VALIDATION_GRID 必须从 TRAIN_GRIDS 中留出。')
    if len(set(EVAL_GRIDS)) != len(EVAL_GRIDS):
        raise ValueError('验证网格与测试网格不能重复。')
    overlap = set(EVAL_GRIDS).intersection(TRAIN_GRIDS)
    if overlap:
        raise ValueError(f'验证/测试网格不能出现在训练集合中: {sorted(overlap)}')

    # ===== Step 1: 密网格求解参考流场 =====
    print('\n' + '=' * 68)
    print(f'Step 1: 密网格参考解 ({NX_DENSE} x {NY_DENSE})')
    print('=' * 68)
    solver_dense = ChannelFlowSIMPLE(NX_DENSE, NY_DENSE, LX, LY, RHO, MU, U_IN)
    solver_dense.solve(max_iter=MAX_ITER, tol=SOLVER_TOL,
                       alpha_u=ALPHA_U, alpha_v=ALPHA_V, alpha_p=ALPHA_P)

    interp_u = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.u)
    interp_v = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.v)
    interp_p = RectBivariateSpline(solver_dense.x, solver_dense.y, solver_dense.p)

    # ===== Step 2-3: 为训练、验证和测试分辨率统一采样并计算离散残差 =====
    all_grids = list(dict.fromkeys(TRAIN_GRIDS + EVAL_GRIDS))
    residual_cases = {}
    print('\n' + '=' * 68)
    print('Step 2-3: 多粗糙度采样 + 完备离散残差计算')
    print('=' * 68)

    for nx, ny in all_grids:
        coarse = ChannelFlowSIMPLE(nx, ny, LX, LY, RHO, MU, U_IN)
        u_sampled = interp_u(coarse.x, coarse.y)
        v_sampled = interp_v(coarse.x, coarse.y)
        p_sampled = interp_p(coarse.x, coarse.y)
        res_u, res_v, res_div = compute_discrete_residual(
            coarse, u_sampled, v_sampled, p_sampled, alpha_u=ALPHA_U
        )
        residual_cases[(nx, ny)] = {
            'solver': coarse,
            'residuals': (res_u, res_v, res_div),
            'sampled': (u_sampled, v_sampled, p_sampled),
        }
        if (nx, ny) in TRAIN_GRIDS:
            role = 'TRAIN'
        elif (nx, ny) == VALIDATION_GRID:
            role = 'VALIDATION'
        else:
            role = 'TEST (held out)'
        print(f'[{role:16s}] grid={nx:2d}x{ny:2d} | '
              f'dx/Lx={coarse.dx / LX:.5f}, dy/Ly={coarse.dy / LY:.5f} | '
              f'max|Ru|={np.max(np.abs(res_u)):.3e}, '
              f'max|Rv|={np.max(np.abs(res_v)):.3e}, '
              f'max|Rdiv|={np.max(np.abs(res_div)):.3e}')

    train_cases = {g: residual_cases[g] for g in TRAIN_GRIDS}
    validation_cases = {VALIDATION_GRID: residual_cases[VALIDATION_GRID]}

    # ===== Step 4: 联合训练。两个新增测试网格不参与训练或验证 =====
    print('\n' + '=' * 68)
    print(f'Step 4: 联合训练，训练分辨率={TRAIN_GRIDS}，验证={VALIDATION_GRID}')
    print(f'训练后独立测试分辨率={TEST_GRIDS}')
    print('=' * 68)
    corrector = NeuralCorrector()
    train_data, val_data = corrector.prepare_datasets(train_cases, validation_cases)
    corrector.train(train_data, validation_data=val_data,
                     epochs=TRAIN_EPOCHS, lr=1.5e-3)
    corrector.save(MODEL_PATH)

    # ===== Step 5-7: 在原验证网格和新增两组网格上独立求解、统计与绘图 =====
    # 每组网格单独输出 3 张图片：残差云图、流场云图、速度剖面对比图。
    print('\n' + '=' * 68)
    print('Step 5-7: 三组未参与训练的分辨率逐一测试')
    print('=' * 68)
    summary_rows = []

    for grid in EVAL_GRIDS:
        nx_eval, ny_eval = grid
        grid_label = f'{nx_eval}x{ny_eval}'
        print('\n' + '-' * 68)
        if grid == VALIDATION_GRID:
            print(f'测试/验证网格 {grid_label}（训练期间只用于验证）')
        else:
            print(f'新增独立测试网格 {grid_label}（未参与训练和验证）')
        print('-' * 68)

        solver_coarse_orig = ChannelFlowSIMPLE(
            nx_eval, ny_eval, LX, LY, RHO, MU, U_IN
        )
        solver_coarse_orig.solve(max_iter=MAX_ITER, tol=SOLVER_TOL,
                                 alpha_u=ALPHA_U, alpha_v=ALPHA_V, alpha_p=ALPHA_P,
                                 verbose=False)

        solver_coarse_corrected = ChannelFlowSIMPLE(
            nx_eval, ny_eval, LX, LY, RHO, MU, U_IN
        )
        res_u_pred, res_v_pred, res_div_pred = corrector.predict(solver_coarse_corrected)
        solver_coarse_corrected.Sp_nn_u = res_u_pred
        solver_coarse_corrected.Sp_nn_v = res_v_pred
        solver_coarse_corrected.Sp_nn_div = res_div_pred
        solver_coarse_corrected.solve(max_iter=MAX_ITER, tol=SOLVER_TOL,
                                      alpha_u=ALPHA_U, alpha_v=ALPHA_V, alpha_p=ALPHA_P,
                                      verbose=False)

        u_ref_on_coarse = interp_u(solver_coarse_orig.x, solver_coarse_orig.y)
        interior = (slice(1, -1), slice(1, -1))
        err_orig = u_ref_on_coarse[interior] - solver_coarse_orig.u[interior]
        err_corr = u_ref_on_coarse[interior] - solver_coarse_corrected.u[interior]
        err_orig_max = float(np.max(np.abs(err_orig)))
        err_corr_max = float(np.max(np.abs(err_corr)))
        err_orig_l2 = float(np.sqrt(np.mean(err_orig ** 2)))
        err_corr_l2 = float(np.sqrt(np.mean(err_corr ** 2)))
        imp_max = (1.0 - err_corr_max / max(err_orig_max, 1e-12)) * 100.0
        imp_l2 = (1.0 - err_corr_l2 / max(err_orig_l2, 1e-12)) * 100.0

        print(f'[Max Error] baseline={err_orig_max:.6e} -> NN={err_corr_max:.6e} '
              f'(变化改善率 {imp_max:.2f}%)')
        print(f'[RMSE]      baseline={err_orig_l2:.6e} -> NN={err_corr_l2:.6e} '
              f'(变化改善率 {imp_l2:.2f}%)')
        summary_rows.append((nx_eval, ny_eval, err_orig_max, err_corr_max,
                             err_orig_l2, err_corr_l2, imp_max, imp_l2))

        res_u_true = residual_cases[grid]['residuals'][0]
        plot_residual_comparison(
            residual_cases[grid]['solver'], res_u_true, res_u_pred,
            OUTPUT_DIR, grid_label
        )
        plot_flow_comparisons(
            solver_dense, solver_coarse_orig, solver_coarse_corrected,
            u_ref_on_coarse, OUTPUT_DIR, grid_label
        )
        plot_velocity_profiles(
            solver_dense, solver_coarse_orig, solver_coarse_corrected,
            u_ref_on_coarse, OUTPUT_DIR, grid_label
        )

    # 输出整洁的测试指标汇总 CSV，方便后续比较不同分辨率。
    import csv
    metrics_path = OUTPUT_DIR / 'test_metrics.csv'
    with metrics_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'nx', 'ny', 'baseline_max_abs_error', 'nn_max_abs_error',
            'baseline_rmse', 'nn_rmse', 'max_error_improvement_percent',
            'rmse_improvement_percent'
        ])
        writer.writerows(summary_rows)
    print(f'\n测试指标汇总已保存: {metrics_path}')
    print(f'全部图片已分别保存到目录: {OUTPUT_DIR.resolve()}')
    print('\n说明：三个测试分辨率固定了同一几何、流体物性和入口条件。')
    print('若要泛化到不同雷诺数/物性/入口速度，应继续将这些物理参数加入网络输入并扩增训练工况。')
