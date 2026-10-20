import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. 物理参数与网格设置
# ==========================================
L = 1.0         # 求解域长度
c = 1.0         # 对流速度
D = 0.2        # 物理扩散系数
u0 = 1.0        # 左边界条件
uL = 0.0        # 右边界条件

N_fine = 200    # 密网格单元数
N_coarse = 5   # 粗网格单元数 (保证 N_fine 是 N_coarse 的整数倍以方便映射)

# ==========================================
# 2. FDM 求解器定义
# ==========================================
def solve_1d_cd_2nd_upwind(N, c, D, u0, uL):
    """密网格：二阶迎风格式求解对流扩散方程"""
    dx = L / N
    x = np.linspace(0, L, N+1)
    A = np.zeros((N-1, N-1))
    b = np.zeros(N-1)

    P1 = c / dx          # 一阶对流系数 (用于第一个内点)
    P2 = c / (2 * dx)    # 二阶对流系数
    Q = D / dx**2        # 中心差分扩散系数

    # 第一个内点 (i=1)，为了模板不越界，降阶使用一阶迎风
    A[0, 0] = P1 + 2*Q
    A[0, 1] = -Q
    b[0] = (P1 + Q) * u0
    
    # 内部点 (i=2 到 N-1)，使用二阶迎风
    for i in range(1, N-1):
        A[i, i] = 3*P2 + 2*Q       # u_i 系数
        A[i, i-1] = -4*P2 - Q      # u_{i-1} 系数
        if i > 1:
            A[i, i-2] = P2         # u_{i-2} 系数
        else:
            b[i] -= P2 * u0        # 边界项移至右端
            
        if i < N-2:
            A[i, i+1] = -Q         # u_{i+1} 系数
        else:
            b[i] += Q * uL         # 边界项移至右端

    u_int = np.linalg.solve(A, b)
    u = np.zeros(N+1)
    u[0], u[-1] = u0, uL
    u[1:-1] = u_int
    return x, u

def solve_1d_cd_1st_upwind(N, c, D, u0, uL, S=None):
    """粗网格：一阶迎风格式求解对流扩散方程 (可外加源项 S)"""
    dx = L / N
    x = np.linspace(0, L, N+1)
    A = np.zeros((N-1, N-1))
    b = np.zeros(N-1)
    if S is None:
        S = np.zeros(N-1)

    P = c / dx
    Q = D / dx**2

    for i in range(N-1):
        A[i, i] = P + 2*Q
        if i > 0:
            A[i, i-1] = -P - Q
        if i < N-2:
            A[i, i+1] = -Q

    # 边界条件结合外加源项
    b[0] = S[0] + (P + Q) * u0
    for i in range(1, N-2):
        b[i] = S[i]
    b[-1] = S[-1] + Q * uL

    # 返回矩阵A和b用于后续计算残差
    u_int = np.linalg.solve(A, b)
    u = np.zeros(N+1)
    u[0], u[-1] = u0, uL
    u[1:-1] = u_int
    return x, u, A, b - S  # 返回纯基础右端项(剔除外加源项以便复用)

# ==========================================
# 3. 生成基准数据并计算残差 (Ground Truth Source)
# ==========================================
# 3.1 求解密网格 (高保真)
x_fine, u_fine = solve_1d_cd_2nd_upwind(N_fine, c, D, u0, uL)

# 3.2 求解粗网格基线 (Base)
x_coarse, u_coarse_base, A_coarse, b_coarse_base = solve_1d_cd_1st_upwind(N_coarse, c, D, u0, uL)

# 3.3 将密网格解投影到粗网格 (空间亚采样)
ratio = N_fine // N_coarse
u_target = u_fine[::ratio]
u_target_int = u_target[1:-1] # 仅取内点参与矩阵运算

# 3.4 提取补偿源项 S_true
# 线性系统 A*u = b + S  =>  S = A*u - b
S_true = A_coarse.dot(u_target_int) - b_coarse_base

# ==========================================
# 4. PyTorch 数据驱动训练
# ==========================================
# 准备训练数据：输入为坐标 x, 输出为残差 S
x_train = torch.tensor(x_coarse[1:-1], dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(S_true, dtype=torch.float32).view(-1, 1)

# 定义简单的多层感知机 (MLP)
class SourceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),         # GELU 对于导数和拟合表现比 ReLU 更好
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

model = SourceNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("开始训练神经网络...")
for epoch in range(1500):
    optimizer.zero_grad()
    predictions = model(x_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6e}")

# ==========================================
# 5. 模型推理与补偿验证
# ==========================================
model.eval()
with torch.no_grad():
    S_ml = model(x_train).numpy().flatten()

# 将预测的源项代回粗网格求解器进行补偿计算
_, u_coarse_comp, _, _ = solve_1d_cd_1st_upwind(N_coarse, c, D, u0, uL, S=S_ml)

# ==========================================
# 6. 结果可视化
# ==========================================
plt.figure(figsize=(12, 5))

# 图1：物理量 u(x) 剖面对比
plt.subplot(1, 2, 1)
plt.plot(x_fine, u_fine, 'k-', linewidth=2, label='Fine Grid (2nd-order upwind)')
plt.plot(x_coarse, u_coarse_base, 'bs--', label='Coarse Base (1st-order upwind)')
plt.plot(x_coarse, u_coarse_comp, 'ro-', label='Coarse ML-Compensated')
plt.title('Velocity Profile u(x)')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True, linestyle=':')

# 图2：源项分布对比
plt.subplot(1, 2, 2)
plt.plot(x_coarse[1:-1], S_true, 'k-', linewidth=2, label='Ground Truth Source $S_{true}$')
plt.plot(x_coarse[1:-1], S_ml, 'r--', linewidth=2, label='ML Predicted Source $S_{ML}$')
plt.title('Compensation Source Term S(x)')
plt.xlabel('x')
plt.ylabel('Source Term S')
plt.legend()
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.show()