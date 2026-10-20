import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_sample(data_dir, sample_idx):
    x_path = os.path.join(data_dir, "Channel_X.npy")
    y_path = os.path.join(data_dir, "Channel_Y.npy")
    q_path = os.path.join(data_dir, "Channel_Q.npy")
    params_path = os.path.join(data_dir, "Channel_Params.npy")

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

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    fig.suptitle(f"Channel Flow PINN Data - Sample {sample_idx}\n$U_{{in}}$={u_in:.4f}, $\\nu$={nu:.6f}", fontsize=14)

    c1 = axes[0].pcolormesh(X, Y, u, shading='auto', cmap='jet')
    axes[0].set_title('Velocity U')
    fig.colorbar(c1, ax=axes[0])

    c2 = axes[1].pcolormesh(X, Y, v, shading='auto', cmap='jet')
    axes[1].set_title('Velocity V')
    fig.colorbar(c2, ax=axes[1])

    c3 = axes[2].pcolormesh(X, Y, p, shading='auto', cmap='jet')
    axes[2].set_title('Pressure P')
    fig.colorbar(c3, ax=axes[2])

    plt.tight_layout()
    save_path = f"sample_{sample_idx}_contour.png"
    plt.savefig(save_path, dpi=150)
    print(f"云图已保存至当前目录: {save_path}")
    # plt.show() # 取消阻塞显示以支持流程自动化

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./channel_data')
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    plot_sample(args.data_dir, args.idx)