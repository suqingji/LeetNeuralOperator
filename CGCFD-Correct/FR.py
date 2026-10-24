import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# --- 问题参数设置 ---
L = 1.0           # 区域长度
a = 1.0           # 波速
N = 40            # 单元数
dx = L / N        # 单元长度
CFL = 0.1         # CFL数
dt = CFL * dx / a # 时间步长
T = 1.0           # 模拟总时间 (跑一整圈)
steps = int(np.ceil(T / dt))
dt = T / steps

# 初始条件
def initial_condition(x):
    return np.sin(2 * np.pi * x)

# --- 1. CG 方法初始化 ---
x_cg = np.linspace(0, L, N, endpoint=False) # 周期边界，不包含最后一个点
# 组装全局质量矩阵 M 和 刚度矩阵 K
M_cg = np.zeros((N, N))
K_cg = np.zeros((N, N))
for i in range(N):
    M_cg[i, i] = 4/6 * dx
    M_cg[i, (i-1)%N] = 1/6 * dx
    M_cg[i, (i+1)%N] = 1/6 * dx
    
    K_cg[i, (i-1)%N] = -a / 2
    K_cg[i, (i+1)%N] = a / 2
M_inv_cg = la.inv(M_cg)
U_cg = initial_condition(x_cg)

# --- 2. DG 方法初始化 (Modal P1) ---
# U_dg[:, 0] 是均值, U_dg[:, 1] 是斜率
x_centers = np.linspace(dx/2, L-dx/2, N)
U_dg = np.zeros((N, 2))
U_dg[:, 0] = initial_condition(x_centers) # 简单起见，取中点值为初始均值
U_dg[:, 1] = 0.0 # 初始斜率近似为0

# --- 3. FR 方法初始化 (Nodal P1) ---
# 高斯求解点位置 (-1/sqrt(3), 1/sqrt(3))
xi_0, xi_1 = -1/np.sqrt(3), 1/np.sqrt(3)
x_fr = np.zeros((N, 2))
x_fr[:, 0] = x_centers + xi_0 * dx/2
x_fr[:, 1] = x_centers + xi_1 * dx/2
U_fr = initial_condition(x_fr)

# 外推系数 (Lagrange 基函数在边界 -1 和 1 的值)
c1 = (1 + np.sqrt(3)) / 2
c2 = (1 - np.sqrt(3)) / 2

# === 右端项计算函数 ===

def rhs_cg(U):
    return -M_inv_cg @ (K_cg @ U)

def rhs_dg(U):
    dU = np.zeros_like(U)
    # 单元边界重构值 (内部: -1, 1)
    u_left_bnd = U[:, 0] - U[:, 1]  # 本单元左边界值
    u_right_bnd = U[:, 0] + U[:, 1] # 本单元右边界值
    # 迎风数值通量
    f_star_left = a * np.roll(u_right_bnd, 1)  # 左侧界面通量 (来自左边单元)
    f_star_right = a * u_right_bnd             # 右侧界面通量 (来自自己)
    
    dU[:, 0] = -1 / dx * (f_star_right - f_star_left)
    dU[:, 1] = -3 / dx * (f_star_right + f_star_left - 2 * a * U[:, 0])
    return dU

def rhs_fr(U):
    dU = np.zeros_like(U)
    u0, u1 = U[:, 0], U[:, 1]
    
    # 边界外推值
    u_L = c1 * u0 + c2 * u1
    u_R = c2 * u0 + c1 * u1
    
    # 迎风数值通量
    f_star_L = a * np.roll(u_R, 1)
    f_star_R = a * u_R
    
    # 物理通量在求解点处的偏导数 dF/dxi
    df_dxi_0 = a * (-np.sqrt(3)/2 * u0 + np.sqrt(3)/2 * u1)
    df_dxi_1 = a * (-np.sqrt(3)/2 * u0 + np.sqrt(3)/2 * u1)
    
    # FR 更新公式 (恢复 DG)
    dU[:, 0] = -2 / dx * (df_dxi_0 - 0.5*(f_star_L - a*u_L) + 0.5*(f_star_R - a*u_R))
    dU[:, 1] = -2 / dx * (df_dxi_1 - 0.5*(f_star_L - a*u_L) + 0.5*(f_star_R - a*u_R))
    return dU

# === TVD-RK3 时间积分 ===
def rk3_step(U, rhs_func, dt):
    U1 = U + dt * rhs_func(U)
    U2 = 0.75 * U + 0.25 * U1 + 0.25 * dt * rhs_func(U1)
    U_new = 1/3 * U + 2/3 * U2 + 2/3 * dt * rhs_func(U2)
    return U_new

for step in range(steps):
    U_cg = rk3_step(U_cg, rhs_cg, dt)
    U_dg = rk3_step(U_dg, rhs_dg, dt)
    U_fr = rk3_step(U_fr, rhs_fr, dt)

# === 绘图结果 ===
plt.figure(figsize=(10, 6))
x_exact = np.linspace(0, L, 200)
plt.plot(x_exact, initial_condition(x_exact - a*T), 'k--', label='Exact Solution')

# CG 绘制 (连续节点)
plt.plot(np.append(x_cg, L), np.append(U_cg, U_cg[0]), 'b-o', markersize=4, label='CG (Linear)')

# DG 绘制 (为了展示间断，在单元内部多取几个点)
x_dg_plot = []
u_dg_plot = []
for i in range(N):
    xi = np.linspace(-1, 1, 5)
    x_dg_plot.extend(x_centers[i] + xi * dx / 2)
    u_dg_plot.extend(U_dg[i, 0] + U_dg[i, 1] * xi)
    # 添加 NaN 使断点不相连
    x_dg_plot.append(np.nan)
    u_dg_plot.append(np.nan)
plt.plot(x_dg_plot, u_dg_plot, 'r-', linewidth=2, label='DG (Modal P1)')

# FR 绘制 (求解点)
plt.plot(x_fr.flatten(), U_fr.flatten(), 'g^', markersize=5, label='FR (Nodal P1 / SPs)')

plt.title(f'1D Advection at T={T:.2f} (Grid: N={N})')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()

# 验证 FR 是否完美等价于 DG
# 将 FR 的 nodal 结果转化为 modal 的均值和斜率
U_fr_mean = 0.5 * (U_fr[:, 0] + U_fr[:, 1])
U_fr_slope = (np.sqrt(3)/2) * (U_fr[:, 1] - U_fr[:, 0])
diff_mean = np.max(np.abs(U_dg[:, 0] - U_fr_mean))
print(f"FR 与 DG 均值的最大差值 (理论应在机器精度级别): {diff_mean:.2e}")