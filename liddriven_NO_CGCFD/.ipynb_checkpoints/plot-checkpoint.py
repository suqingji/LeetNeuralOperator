import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_sample(data_dir, sample_idx):
    x_path = os.path.join(data_dir, "Cavity_X.npy")
    y_path = os.path.join(data_dir, "Cavity_Y.npy")
    q_path = os.path.join(data_dir, "Cavity_Q.npy")
    params_path = os.path.join(data_dir, "Cavity_Params.npy")

    X_data = np.load(x_path)
    Y_data = np.load(y_path)
    Q_data = np.load(q_path)
    Params_data = np.load(params_path)

    num_samples = X_data.shape[0]
    if sample_idx < 0 or sample_idx >= num_samples:
        print(f"错误：样本索引 {sample_idx} 超出范围 (0 到 {num_samples-1})")
        return

    X, Y = X_data[sample_idx], Y_data[sample_idx]
    u, v, p = Q_data[sample_idx, 0], Q_data[sample_idx, 1], Q_data[sample_idx, 2]
    u_in, nu = Params_data[sample_idx]

    # 计算速度大小 Magnitude: mag = sqrt(u^2 + v^2)
    mag = np.sqrt(u**2 + v**2)

    # 创建 2x2 子图布局
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 9))
    fig.suptitle(f"Lid-Driven Cavity PINN Data - Sample {sample_idx}\n$U_{{lid}}$={u_in:.4f}, $\\nu$={nu:.6f}", fontsize=14)

    # 1. Top-Left: Velocity U
    c00 = axes[0, 0].pcolormesh(X, Y, u, shading='auto', cmap='jet')
    axes[0, 0].set_title('Velocity U')
    axes[0, 0].set_aspect('equal')
    fig.colorbar(c00, ax=axes[0, 0])

    # 2. Top-Right: Velocity V
    c01 = axes[0, 1].pcolormesh(X, Y, v, shading='auto', cmap='jet')
    axes[0, 1].set_title('Velocity V')
    axes[0, 1].set_aspect('equal')
    fig.colorbar(c01, ax=axes[0, 1])

    # 3. Bottom-Left: Velocity Magnitude |U|
    c10 = axes[1, 0].pcolormesh(X, Y, mag, shading='auto', cmap='jet')
    axes[1, 0].set_title('Velocity Magnitude |U|')
    axes[1, 0].set_aspect('equal')
    fig.colorbar(c10, ax=axes[1, 0])

    # 4. Bottom-Right: Pressure P
    c11 = axes[1, 1].pcolormesh(X, Y, p, shading='auto', cmap='jet')
    axes[1, 1].set_title('Pressure P')
    axes[1, 1].set_aspect('equal')
    fig.colorbar(c11, ax=axes[1, 1])

    plt.tight_layout()
    save_path = f"sample_{sample_idx}_contour.png"
    plt.savefig(save_path, dpi=150)
    print(f"云图已保存至当前目录: {save_path}")

if __name__ == "__main__":
    target_file = "./sample_0_contour.png"
    if os.path.exists(target_file):
        os.remove(target_file)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./cavity_data')
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    plot_sample(args.data_dir, args.idx)