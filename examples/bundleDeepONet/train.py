import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 物理网络架构定义 (DeepONet)
# ==========================================
class DeepONet(nn.Module):
    def __init__(self, branch_in_dim, trunk_in_dim, width=128, p_dim=64):
        """
        DeepONet 神经网络架构
        :param branch_in_dim: Branch Net 输入维度 (即物理参数的个数)
        :param trunk_in_dim: Trunk Net 输入维度 (位置坐标 1 维 + 变量标签 One-hot 6 维 = 7 维)
        :param width: 隐层神经元数量
        :param p_dim: 基函数/特征系数的数量 (Branch 和 Trunk 输出的对齐维度)
        """
        super(DeepONet, self).__init__()
        
        # 分支网络 (Branch Net) - 提取全局物理参数特征
        self.branch = nn.Sequential(
            nn.Linear(branch_in_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, p_dim)
        )
        
        # 主干网络 (Trunk Net) - 提取局部连续时空域及变量特征
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, p_dim)
        )
        
        # 可学习的偏置项
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, trunk_input):
        # 分别前向传播
        b_out = self.branch(branch_input)  # [batch_size, p_dim]
        t_out = self.trunk(trunk_input)    # [batch_size, p_dim]
        
        # 点乘融合 (内积) 加偏置
        output = torch.sum(b_out * t_out, dim=-1, keepdim=True) + self.b0
        return output

# ==========================================
# 2. 自定义数据集类 (Dataset)
# ==========================================
class DeepONetDataset(Dataset):
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.tensor(branch_data, dtype=torch.float32)
        self.trunk_data = torch.tensor(trunk_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        return self.branch_data[idx], self.trunk_data[idx], self.target_data[idx]

# ==========================================
# 3. 数据读取与预处理函数
# ==========================================
def load_and_preprocess_data(param_file, base_dir):
    """
    读取 param.txt 和各物理量文件夹下的数据文件，并转换为多变量标量数据集
    """
    # 6个物理量的映射，使用 One-hot 编码 (维度为6)
    var_map = {
        'L1-U': 0, 'L1-V': 1, 'L1-W': 2,
        'L2-U': 3, 'L2-V': 4, 'L2-W': 5
    }
    
    # 读取主参数文件 param.txt (假设以空格或逗号分隔)
    # N 行代表 N 组参数
    params = np.loadtxt(param_file)
    if len(params.shape) == 1:
        params = params.reshape(1, -1)  # 处理只有1组参数的边界情况
    N = params.shape[0]
    
    all_branch_inputs = []
    all_trunk_inputs = []
    all_targets = []
    
    print(f"开始读取数据，总计检测到 {N} 组物理参数。")
    
    # 遍历每组物理参数
    for i in range(N):
        current_param = params[i]
        # 假设文件夹命名规则为 group_0, group_1, ..., group_N-1
        # 如果您的实际文件夹名是以 1 开始或有其他前缀，请在此处修改
        group_folder = os.path.join(base_dir, f"group_{i}")
        
        if not os.path.exists(group_folder):
            print(f"警告: 文件夹 {group_folder} 不存在，已跳过。")
            continue
            
        # 遍历该组参数下的6个数据文件
        for var_name, var_idx in var_map.items():
            file_path = os.path.join(group_folder, f"{var_name}.txt")
            
            if not os.path.exists(file_path):
                # 兼容不带扩展名的情况
                file_path = os.path.join(group_folder, var_name)
                if not os.path.exists(file_path):
                    continue
            
            # 读取数据，跳过第一行 (第一行是 tag 文本)
            # numpy loadtxt 可以通过 skiprows=1 轻松跳过表头
            try:
                data = np.loadtxt(file_path, skiprows=1)
            except Exception as e:
                print(f"读取文件 {file_path} 出错: {e}")
                continue
                
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
                
            # 第一列：无量纲坐标；第二列：物理量预测目标值
            coords = data[:, 0]
            targets = data[:, 1]
            
            # 为当前文件中的每一个坐标点构建训练样本
            for c, t in zip(coords, targets):
                # 1. Branch 输入：当前组的物理参数
                all_branch_inputs.append(current_param)
                
                # 2. Trunk 输入：无量纲坐标 + 6维的变量 One-hot 编码
                one_hot = np.zeros(6)
                one_hot[var_idx] = 1.0
                trunk_in = np.hstack(([c], one_hot))  # 组合成 7 维向量
                all_trunk_inputs.append(trunk_in)
                
                # 3. 目标输出值
                all_targets.append([t])
                
    return (np.array(all_branch_inputs), 
            np.array(all_trunk_inputs), 
            np.array(all_targets))

# ==========================================
# 4. 主训练主控流程
# ==========================================
def main():
    # ---- 路径配置 (请根据实际情况修改) ----
    PARAM_FILE = "param.txt"      # 参数文件路径
    BASE_DIR = "./data"           # 存放各个组文件夹的根目录
    MODEL_SAVE_PATH = "deeponet_multivar_model.pt"
    
    # ---- 超参数设置 ----
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-3
    EPOCHS = 200
    HIDDEN_WIDTH = 128
    P_DIM = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前训练设备: {DEVICE}")

    # 1. 加载并转换数据
    # 注意：在实际运行前，请确保 PARAM_FILE 和 BASE_DIR 路径下有对应的仿真数据
    try:
        X_branch, X_trunk, Y = load_and_preprocess_data(PARAM_FILE, BASE_DIR)
    except FileNotFoundError:
        print("未检测到真实数据文件，生成一组模拟数据用于代码结构验证...")
        # 模拟生成 5 组参数，每组参数有 4 个维度
        X_branch = np.random.randn(3000, 4)
        # 模拟 Trunk 输入：1维坐标 + 6维 one-hot
        X_trunk = np.hstack((np.random.uniform(0, 10, (3000, 1)), np.eye(6)[np.random.randint(0, 6, 3000)]))
        Y = np.random.randn(3000, 1)
        
    print(f"数据处理完成！样本总数: {len(Y)}")
    print(f"Branch 输入形状: {X_branch.shape}, Trunk 输入形状: {X_trunk.shape}, 输出形状: {Y.shape}")

    # 2. 构建数据加载器
    dataset = DeepONetDataset(X_branch, X_trunk, Y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 初始化网络模型
    branch_in_dim = X_branch.shape[1]
    trunk_in_dim = X_trunk.shape[1]  # 应该是 7 (1 + 6)
    
    model = DeepONet(branch_in_dim=branch_in_dim, 
                     trunk_in_dim=trunk_in_dim, 
                     width=HIDDEN_WIDTH, 
                     p_dim=P_DIM).to(DEVICE)

    # 4. 定义损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # 5. 训练循环
    model.train()
    print("开始训练网络...")
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for batch_idx, (b_in, t_in, target) in enumerate(dataloader):
            b_in, t_in, target = b_in.to(DEVICE), t_in.to(DEVICE), target.to(DEVICE)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播
            prediction = model(b_in, t_in)
            
            # 计算均方误差损失
            loss = criterion(prediction, target)
            
            # 反向传播及优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * b_in.size(0)
            
        epoch_loss /= len(dataset)
        scheduler.step(epoch_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{EPOCHS}] - Loss (MSE): {epoch_loss:.6f}")

    # 6. 保存训练好的模型权重及网络结构
    torch.save({
        'model_state_dict': model.state_dict(),
        'branch_in_dim': branch_in_dim,
        'trunk_in_dim': trunk_in_dim,
        'hidden_width': HIDDEN_WIDTH,
        'p_dim': P_DIM
    }, MODEL_SAVE_PATH)
    print(f"模型已成功训练并保存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()