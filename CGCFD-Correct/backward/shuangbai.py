import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义双摆系统的物理参数
# ==========================================
m1 = 1.0    # 第一杆质量 (kg)
m2 = 1.0    # 第二杆质量 (kg)
l1 = 1.0    # 第一杆长度 (m)
l2 = 1.0    # 第二杆长度 (m)
g = 9.81    # 重力加速度 (m/s^2)

# ==========================================
# 2. 定义状态方程 (基于拉格朗日方程推导)
# ==========================================
def double_pendulum_ode(t, y):
    """
    y = [theta1, omega1, theta2, omega2]
    """
    theta1, omega1, theta2, omega2 = y
    
    delta = theta1 - theta2
    
    # 简化后的加速度分母项
    den1 = l1 * (2.0 * m1 + m2 - m2 * np.cos(2.0 * theta1 - 2.0 * theta2))
    den2 = l2 * (2.0 * m1 + m2 - m2 * np.cos(2.0 * theta1 - 2.0 * theta2))
    
    # 角加速度 alpha1
    num1 = -g * (2.0 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2.0 * theta2) - 2.0 * np.sin(delta) * m2 * (omega2**2 * l2 + omega1**2 * l1 * np.cos(delta))
    alpha1 = num1 / den1
    
    # 角加速度 alpha2
    num2 = 2.0 * np.sin(delta) * (omega1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + omega2**2 * l2 * m2 * np.cos(delta))
    alpha2 = num2 / den2
    
    return [omega1, alpha1, omega2, alpha2]

# ==========================================
# 3. 初始条件与时间跨度
# ==========================================
# 初始角度（弧度）和初速度
theta1_init = np.pi / 2  # 90度
omega1_init = 0.0
theta2_init = np.pi / 2  # 90度
omega2_init = 0.0

y0 = [theta1_init, omega1_init, theta2_init, omega2_init]

t_start = 0.0
t_end = 20.0
t_eval = np.linspace(t_start, t_end, 1000)

# ==========================================
# 4. 数值求解微分方程
# ==========================================
sol = solve_ivp(double_pendulum_ode, [t_start, t_end], y0, t_eval=t_eval, method='RK45')

# ==========================================
# 5. 结果可视化
# ==========================================
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[0], label='Theta 1 (rad)', color='b')
plt.plot(sol.t, sol.y[2], label='Theta 2 (rad)', color='r')
plt.title('Double Pendulum Dynamics (Lagrangian Formulation)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(sol.t, sol.y[1], label='Omega 1 (rad/s)', color='cyan')
plt.plot(sol.t, sol.y[3], label='Omega 2 (rad/s)', color='magenta')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()