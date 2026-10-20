import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_sample(data_dir, sample_idx):
    # 1. 定义数据路径
    x_path = os.path.join(data_dir, "Channel_X.npy")
    y_path = os.path.join(data_dir, "Channel_Y.npy")
    q_path = os.path.join(data_dir, "Channel_Q.npy")
    params_path = os.path.join(data_dir, "Channel_Params.npy")

    # 2. 检查文件是否存在
    if not all(map(os.path.exists, [x_path, y_path, q_path, params_path])):
        print(f"错误：在 {data_dir} 找不到完整的数据文件，请确认路径。")
        return

    # 3. 加载数据
    X_data = np.load(x_path)
    Y_data = np.load(y_path)
    Q_data = np.load(q_path)
    Params_data = np.load(params_path)

    num_samples = X_data.shape[0]
    if sample_idx < 0 or sample_idx >= num_samples:
        print(f"错误：样本索引 {sample_idx} 超出范围 (0 到 {num_samples-1})")
        return

    # 4. 提取当前样本的信息
    X = X_data[sample_idx]
    Y = Y_data[sample_idx]
    u = Q_data[sample_idx, 0]
    v = Q_data[sample_idx, 1]
    p = Q_data[sample_idx, 2]
    
    u_in, nu = Params_data[sample_idx]

    # 输出样本参数
    print(f"成功加载第 {sample_idx} 号样本：")
    print(f"入口速度 (U_in) : {u_in:.4f} m/s")
    print(f"运动粘度 (nu)   : {nu:.6f} m^2/s")

    # 5. 开始绘制云图
    # 由于槽道尺寸为 10x1，长宽比很大，使用纵向排列的3个子图展示
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    fig.suptitle(f"Channel Flow PINN Data - Sample {sample_idx}\n$U_{{in}}$={u_in:.4f}, $\\nu$={nu:.6f}", fontsize=14)

    # 绘制 u 速度场
    c1 = axes[0].pcolormesh(X, Y, u, shading='auto', cmap='jet')
    axes[0].set_title('Velocity U')
    axes[0].set_ylabel('Y')
    fig.colorbar(c1, ax=axes[0])

    # 绘制 v 速度场
    c2 = axes[1].pcolormesh(X, Y, v, shading='auto', cmap='jet')
    axes[1].set_title('Velocity V')
    axes[1].set_ylabel('Y')
    fig.colorbar(c2, ax=axes[1])

    # 绘制 p 压力场
    c3 = axes[2].pcolormesh(X, Y, p, shading='auto', cmap='jet')
    axes[2].set_title('Pressure P')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    fig.colorbar(c3, ax=axes[2])

    plt.tight_layout()
    
    # 保存或展示
    save_path = f"sample_{sample_idx}_contour.png"
    plt.savefig(save_path, dpi=150)
    print(f"云图已保存至当前目录: {save_path}")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PINN Channel Flow Data")
    parser.add_argument('--data_dir', type=str, default='./channel_data', help='存放 npy 数据的目录')
    parser.add_argument('--idx', type=int, default=2, help='想要绘制的样本索引编号 (默认: 0)')
    
    args = parser.parse_args()
    plot_sample(args.data_dir, args.idx)