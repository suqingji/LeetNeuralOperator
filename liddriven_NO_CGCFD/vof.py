import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 基于VOF (Volume of Fluid) 方法的旋转涡 (Single Vortex / Rider-Kothe Vortex) 问题求解
# 格式：向量化高效一阶迎风格式 (First-Order Upwind Scheme)
# 描述：
# 速度场由旋转涡流场给出：
#   u(x, y, t) = - sin^2(pi * x) * sin(2 * pi * y) * cos(pi * t / T)
#   v(x, y, t) =   sin(2 * pi * x) * sin^2(pi * y) * cos(pi * t / T)
# 在 t = T/2 时流场达到最大拉伸，在 t = T 时流体界面理论上应复原到初始形态。
# ==============================================================================

def init_vortex_problem(N=128, L=1.0):
    """初始化网格与旋转涡界面 (偏心圆)"""
    dx = L / N
    dy = L / N
    
    # 网格中心坐标
    xc = np.linspace(dx / 2.0, L - dx / 2.0, N)
    yc = np.linspace(dy / 2.0, L - dy / 2.0, N)
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    
    # 界面初始形态：位于 (0.5, 0.75)，半径 R = 0.15 的圆
    cx, cy, R = 0.5, 0.75, 0.15
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    alpha = np.zeros((N, N))
    alpha[dist <= R - dx / 2.0] = 1.0
    mask_boundary = (dist > R - dx / 2.0) & (dist < R + dx / 2.0)
    alpha[mask_boundary] = (R + dx / 2.0 - dist[mask_boundary]) / dx
    alpha = np.clip(alpha, 0.0, 1.0)
                
    return X, Y, dx, dy, alpha

def get_staggered_velocity(N, dx, dy, t, T):
    """计算交错网格上的面心速度 (严格满足离散无散条件 div u = 0)"""
    # u 面 (N+1, N)
    xf_u = np.linspace(0.0, 1.0, N + 1)
    yc_u = np.linspace(dy / 2.0, 1.0 - dy / 2.0, N)
    Xf_u, Yc_u = np.meshgrid(xf_u, yc_u, indexing='ij')
    
    # v 面 (N, N+1)
    xc_v = np.linspace(dx / 2.0, 1.0 - dx / 2.0, N)
    yf_v = np.linspace(0.0, 1.0, N + 1)
    Xc_v, Yf_v = np.meshgrid(xc_v, yf_v, indexing='ij')
    
    # 时间调制因子
    g_t = np.cos(np.pi * t / T)
    
    # 旋转涡速度场公式
    U_face = - (np.sin(np.pi * Xf_u) ** 2) * np.sin(2.0 * np.pi * Yc_u) * g_t
    V_face =   np.sin(2.0 * np.pi * Xc_v) * (np.sin(np.pi * Yf_v) ** 2) * g_t
    
    return U_face, V_face

def upwind_vof_step_vectorized(alpha, U_face, V_face, dx, dy, dt, N):
    """
    向量化一阶迎风格式 (1st-Order Upwind Scheme) 更新 VOF 体积分数 alpha
    """
    # 计算 x 方向面通量 Flux_x (N+1, N)
    Flux_x = np.zeros((N + 1, N))
    u_inner = U_face[1:N, :]
    alpha_left = alpha[:-1, :]
    alpha_right = alpha[1:, :]
    Flux_x[1:N, :] = np.where(u_inner >= 0, u_inner * alpha_left, u_inner * alpha_right)
    
    # 计算 y 方向面通量 Flux_y (N, N+1)
    Flux_y = np.zeros((N, N + 1))
    v_inner = V_face[:, 1:N]
    alpha_bottom = alpha[:, :-1]
    alpha_top = alpha[:, 1:]
    Flux_y[:, 1:N] = np.where(v_inner >= 0, v_inner * alpha_bottom, v_inner * alpha_top)
    
    # 有限体积保守型更新: alpha^{n+1} = alpha^n - dt/dx * (Flux_e - Flux_w) - dt/dy * (Flux_n - Flux_s)
    dfx = Flux_x[1:, :] - Flux_x[:-1, :]
    dfy = Flux_y[:, 1:] - Flux_y[:, :-1]
    
    alpha_new = alpha - (dt / dx) * dfx - (dt / dy) * dfy
    return np.clip(alpha_new, 0.0, 1.0)

def run_simulation():
    N = 128             # 网格数量 N x N
    L = 1.0             # 计算域尺寸
    T = 2.0             # 旋转总周期
    CFL = 0.35          # CFL 极值约束
    
    X, Y, dx, dy, alpha_0 = init_vortex_problem(N, L)
    
    # 最大速度幅度为 1.0
    max_u = 1.0
    dt = CFL * dx / max_u
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps    # 调整 dt 使其整除 T
    
    print(f"================ VOF 旋转涡模拟 (向量化一阶迎风格式) ================")
    print(f"网格分辨率: {N}x{N}, 域大小: [{L}x{L}]")
    print(f"模拟周期 T: {T}s, 时间步数: {n_steps}, 时间步长 dt: {dt:.6f}s")
    
    alpha = np.copy(alpha_0)
    
    # 记录关键时刻形态
    snap_steps = {
        0: np.copy(alpha_0),
        n_steps // 2: None,
        n_steps: None
    }
    
    t = 0.0
    for step in range(1, n_steps + 1):
        U_face, V_face = get_staggered_velocity(N, dx, dy, t, T)
        alpha = upwind_vof_step_vectorized(alpha, U_face, V_face, dx, dy, dt, N)
        t += dt
        
        if step == n_steps // 2:
            snap_steps[n_steps // 2] = np.copy(alpha)
            print(f"已完成 t = T/2 ({t:.3f}s) 阶段计算 (流场拉伸拉细形态)")
            
        if step % (n_steps // 5) == 0 or step == n_steps:
            print(f"进度: Step {step}/{n_steps} (t = {t:.3f}s / {T}s)")
            
    snap_steps[n_steps] = np.copy(alpha)
    
    # 结果绘制与保存
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # t = 0
    c0 = axes[0].contourf(X, Y, snap_steps[0], levels=np.linspace(0, 1, 11), cmap='Blues')
    axes[0].contour(X, Y, snap_steps[0], levels=[0.5], colors='red', linewidths=1.5)
    axes[0].set_title('Initial Interface (t = 0)')
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    
    # t = T/2
    half_step = n_steps // 2
    c1 = axes[1].contourf(X, Y, snap_steps[half_step], levels=np.linspace(0, 1, 11), cmap='Blues')
    axes[1].contour(X, Y, snap_steps[half_step], levels=[0.5], colors='red', linewidths=1.5)
    axes[1].set_title(f'Max Deformation (t = T/2 = {T/2:.1f}s)')
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    
    # t = T
    c2 = axes[2].contourf(X, Y, snap_steps[n_steps], levels=np.linspace(0, 1, 11), cmap='Blues')
    axes[2].contour(X, Y, snap_steps[n_steps], levels=[0.5], colors='red', linewidths=1.5)
    axes[2].contour(X, Y, snap_steps[0], levels=[0.5], colors='black', linestyles='--', linewidths=1.2)
    axes[2].set_title(f'Reconstructed (t = T = {T:.1f}s)\n[Dotted: Initial Boundary]')
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])
    fig.colorbar(c0, cax=cbar_ax, orientation='horizontal', label='Volume Fraction (alpha)')
    
    fig.suptitle("VOF Simulation of Single Vortex Problem (1st-Order Upwind Scheme)", fontsize=14)
    save_path = "vof_vortex_upwind.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果云图已保存至当前目录: {save_path}")

if __name__ == "__main__":
    run_simulation()