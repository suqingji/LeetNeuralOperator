import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ================= 1. 参数设置 =================
N = 64                  # 网格数 N x N
L = 1.0                 # 区域长度
dx = L / N
dy = L / N
x = np.linspace(dx/2, L-dx/2, N)
y = np.linspace(dy/2, L-dy/2, N)
X, Y = np.meshgrid(x, y, indexing='ij')

omega = 2.0 * np.pi     # 旋转角速度
T_end = 1.0             # 旋转一圈的时间
dt = 0.005              # 时间步长
n_steps = int(T_end / dt)

# ================= 2. 初始化流场与界面 =================
# 初始条件：一个偏心的圆形液滴
R0 = 0.15
cx, cy = 0.5, 0.75
alpha_init = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        dist = np.sqrt((X[i,j] - cx)**2 + (Y[i,j] - cy)**2)
        if dist < R0:
            alpha_init[i,j] = 1.0
        # 边界做微小平滑以避免极端的阶跃误差
        elif dist < R0 + dx:
            alpha_init[i,j] = (R0 + dx - dist) / dx 

# 定义交错网格上的速度场 (解析解, 严格散度为0)
U_face = np.zeros((N+1, N)) # x方向面心速度
V_face = np.zeros((N, N+1)) # y方向面心速度
for i in range(N+1):
    for j in range(N):
        xf = i * dx
        yc = (j + 0.5) * dy
        U_face[i, j] = -omega * (yc - 0.5)

for i in range(N):
    for j in range(N+1):
        xc = (i + 0.5) * dx
        yf = j * dy
        V_face[i, j] = omega * (xc - 0.5)

# ================= 3. 代数方法：隐式一阶迎风 (Implicit Upwind) =================
def solve_algebraic_implicit(alpha_n):
    # 构造稀疏矩阵 A 和右端项 b
    # 方程: (V/dt)*alpha_P^{n+1} + sum(F_f * alpha_f^{n+1}) = (V/dt)*alpha_P^n
    num_cells = N * N
    row = []
    col = []
    data = []
    b = np.zeros(num_cells)
    
    vol_dt = (dx * dy) / dt
    
    def get_idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            p_idx = get_idx(i, j)
            b[p_idx] = vol_dt * alpha_n[i, j]
            
            a_p = vol_dt
            
            # West face
            u_w = U_face[i, j]
            flux_w = u_w * dy
            if flux_w > 0: # 出流
                if i > 0:
                    a_p += flux_w
                    row.append(p_idx); col.append(get_idx(i-1, j)); data.append(-flux_w)
            else: # 入流
                a_p += flux_w
                if i > 0:
                    row.append(p_idx); col.append(get_idx(i-1, j)); data.append(-flux_w)
            
            # East face
            u_e = U_face[i+1, j]
            flux_e = u_e * dy
            if flux_e > 0:
                a_p += flux_e
            else:
                if i < N-1:
                    row.append(p_idx); col.append(get_idx(i+1, j)); data.append(flux_e)
                    
            # South face
            v_s = V_face[i, j]
            flux_s = v_s * dx
            if flux_s > 0:
                if j > 0:
                    a_p += flux_s
                    row.append(p_idx); col.append(get_idx(i, j-1)); data.append(-flux_s)
            else:
                a_p += flux_s
                if j > 0:
                    row.append(p_idx); col.append(get_idx(i, j-1)); data.append(-flux_s)
                    
            # North face
            v_n = V_face[i, j+1]
            flux_n = v_n * dx
            if flux_n > 0:
                a_p += flux_n
            else:
                if j < N-1:
                    row.append(p_idx); col.append(get_idx(i, j+1)); data.append(flux_n)
                    
            row.append(p_idx)
            col.append(p_idx)
            data.append(a_p)
            
    A = sp.csr_matrix((data, (row, col)), shape=(num_cells, num_cells))
    # 求解线性方程组 Ax = b
    alpha_next_flat = spla.spsolve(A, b)
    return alpha_next_flat.reshape((N, N))

# ================= 4. 几何方法：显式近似供受体 (Explicit Geometric SLIC) =================
# 这里使用简单的 1D 算子分裂（Operator Splitting）和 Donor-Acceptor 几何通量近似
def solve_geometric_explicit(alpha_n):
    alpha = np.copy(alpha_n)
    
    # X 方向 Sweep
    alpha_star = np.copy(alpha)
    for i in range(N):
        for j in range(N):
            u_w = U_face[i, j]
            u_e = U_face[i+1, j]
            # 计算几何通量 (简单迎风结合界面判断)
            # 为了保持锐利，如果供体网格alpha>0.5，则认为全流出/流入；否则为0
            # 这是一种最简单的几何追踪（Noh & Woodward SLIC 思想）
            flux_in = 0; flux_out = 0
            if u_w > 0 and i > 0:
                f_vol = u_w * dt / dx
                flux_in = min(alpha[i-1, j], f_vol) if alpha[i-1,j]>0.1 else 0
            elif u_w < 0:
                f_vol = -u_w * dt / dx
                flux_out = min(alpha[i, j], f_vol) if alpha[i,j]>0.1 else 0
                
            if u_e > 0:
                f_vol = u_e * dt / dx
                flux_out += min(alpha[i, j], f_vol) if alpha[i,j]>0.1 else 0
            elif u_e < 0 and i < N-1:
                f_vol = -u_e * dt / dx
                flux_in += min(alpha[i+1, j], f_vol) if alpha[i+1,j]>0.1 else 0
                
            alpha_star[i, j] = alpha[i, j] + flux_in - flux_out
            alpha_star[i, j] = np.clip(alpha_star[i, j], 0, 1)

    # Y 方向 Sweep
    alpha_next = np.copy(alpha_star)
    for i in range(N):
        for j in range(N):
            v_s = V_face[i, j]
            v_n = V_face[i, j+1]
            
            flux_in = 0; flux_out = 0
            if v_s > 0 and j > 0:
                f_vol = v_s * dt / dy
                flux_in = min(alpha_star[i, j-1], f_vol) if alpha_star[i,j-1]>0.1 else 0
            elif v_s < 0:
                f_vol = -v_s * dt / dy
                flux_out = min(alpha_star[i, j], f_vol) if alpha_star[i,j]>0.1 else 0
                
            if v_n > 0:
                f_vol = v_n * dt / dy
                flux_out += min(alpha_star[i, j], f_vol) if alpha_star[i,j]>0.1 else 0
            elif v_n < 0 and j < N-1:
                f_vol = -v_n * dt / dy
                flux_in += min(alpha_star[i, j+1], f_vol) if alpha_star[i,j+1]>0.1 else 0
                
            alpha_next[i, j] = alpha_star[i, j] + flux_in - flux_out
            alpha_next[i, j] = np.clip(alpha_next[i, j], 0, 1)
            
    return alpha_next

# ================= 5. 主时间循环与求解 =================
alpha_alg = np.copy(alpha_init)
alpha_geo = np.copy(alpha_init)

print(f"开始求解... 共 {n_steps} 步")
for step in range(n_steps):
    alpha_alg = solve_algebraic_implicit(alpha_alg)
    alpha_geo = solve_geometric_explicit(alpha_geo)
    
    if (step + 1) % 50 == 0:
        print(f"已完成步数: {step + 1}/{n_steps}")

# ================= 6. 结果可视化 =================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 初始状态
c0 = axes[0].contourf(X, Y, alpha_init, levels=np.linspace(0, 1, 11), cmap='Blues')
axes[0].set_title('Initial Condition (t=0)')
axes[0].set_aspect('equal')

# 代数隐式方法结果
c1 = axes[1].contourf(X, Y, alpha_alg, levels=np.linspace(0, 1, 11), cmap='Blues')
axes[1].set_title('Implicit Algebraic VOF (t=1.0)\n(Numerical Diffusion)')
axes[1].set_aspect('equal')

# 几何显式方法结果
c2 = axes[2].contourf(X, Y, alpha_geo, levels=np.linspace(0, 1, 11), cmap='Blues')
axes[2].set_title('Explicit Geometric VOF (t=1.0)\n(Sharp Interface)')
axes[2].set_aspect('equal')

plt.colorbar(c1, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.1)
plt.show()