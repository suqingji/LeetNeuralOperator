import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# ==========================================
# 1. 数据集定义 (Dataset)
# ==========================================
class CFDDataset(Dataset):
    def __init__(self, data_matrix, n_inputs):
        """
        data_matrix: numpy array, 形状为 [样本数, n + m]
        n_inputs: 整数, 前 n 个特征作为输入
        """
        # 提取前 n 列作为输入 X，剩下的作为输出 Y
        self.X = torch.tensor(data_matrix[:, :n_inputs], dtype=torch.float32)
        self.Y = torch.tensor(data_matrix[:, n_inputs:], dtype=torch.float32)
        
        # 建议对物理数据进行归一化/标准化处理（这里提供简单的 Min-Max 记录用于后续还原）
        # 实际工程中，建议在传入 Dataset 前使用 sklearn.preprocessing 处理好
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================
# 2. 神经算子/代理模型架构 (Surrogate Model)
# ==========================================
class CFDSurrogate(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=5):
        """
        input_dim: 输入特征维度 (n)
        output_dim: 输出特征维度 (m)
        hidden_dim: 隐藏层神经元个数
        num_layers: 网络的总层数
        """
        super(CFDSurrogate, self).__init__()
        
        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        # GELU 或 SiLU 激活函数在物理场拟合中通常比 ReLU 表现更平滑、更好
        layers.append(nn.GELU()) 

        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            
        # 输出层 (通常不加激活函数，直接输出实数预测值)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. 训练主控函数 (Trainer)
# ==========================================
def train_cfd_surrogate(data_matrix, n_inputs, epochs=100, batch_size=64, lr=1e-3, device='cuda'):
    """
    接收数据矩阵并完成模型的训练
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 实例化数据集
    dataset = CFDDataset(data_matrix, n_inputs)
    m_outputs = data_matrix.shape[1] - n_inputs
    
    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = CFDSurrogate(input_dim=n_inputs, output_dim=m_outputs).to(device)
    criterion = nn.MSELoss() # 均方误差，流场预测常用
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度器 (可选：当损失平缓时降低学习率)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)

        # 验证循环
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_Y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print("Training Complete!")
    return model

# ==========================================
# 4. 你的执行入口 (How to use)
# ==========================================
if __name__ == "__main__":
    # --- 假设你的数据情况 ---
    n = 10  # 输入维度 (例如：边界条件、雷诺数、马赫数、几何参数等)
    m = 100 # 输出维度 (例如：展平后的流场压力、速度节点分布)
    total_samples = 5000 # 你的样本总数
    
    # ⚠️ 在这里替换成你真实的数据加载逻辑 ⚠️
    # 例如: my_data = np.loadtxt("cfd_data.csv", delimiter=",")
    # 或者: my_data = np.load("cfd_data.npy")
    
    # 这里用随机数据模拟你的 [n + m] 矩阵
    dummy_data = np.random.rand(total_samples, n + m) 
    
    # 启动训练
    trained_model = train_cfd_surrogate(
        data_matrix=dummy_data, 
        n_inputs=n, 
        epochs=100, 
        batch_size=128, 
        lr=0.001,
        device='cuda' # 如果没有GPU会自动回退到CPU
    )
    
    # 保存模型
    torch.save(trained_model.state_dict(), "cfd_surrogate_model.pth")
    print("模型已保存为 cfd_surrogate_model.pth")
