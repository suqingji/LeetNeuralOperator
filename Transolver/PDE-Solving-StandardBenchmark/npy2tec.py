import numpy as np
import os
import argparse

def convert_to_tecplot(data_dir, output_dir, sample_indices):
    # 构建完整的数据文件路径
    x_path = os.path.join(data_dir, "Pipe_X.npy")
    y_path = os.path.join(data_dir, "Pipe_Y.npy")
    q_path = os.path.join(data_dir, "Pipe_Q.npy")
    
    print("正在加载 numpy 数据...")
    X_data = np.load(x_path)
    Y_data = np.load(y_path)
    Q_data = np.load(q_path)
    
    num_samples, nx, ny = X_data.shape
    _, num_channels, _, _ = Q_data.shape
    
    print(f"数据加载成功。总样本数: {num_samples}，网格维度: {nx}x{ny}，Q通道数: {num_channels}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx in sample_indices:
        if idx < 0 or idx >= num_samples:
            print(f"跳过：样本索引 {idx} 超出范围 (0 到 {num_samples - 1})")
            continue
            
        out_filename = os.path.join(output_dir, f"pipe_sample_{idx}.dat")
        print(f"正在导出第 {idx} 个样本 -> {out_filename}")
        
        # 提取当前样本，并将其展平为一维数组 (按 C-contiguous 顺序)
        X_flat = X_data[idx].flatten()
        Y_flat = Y_data[idx].flatten()
        
        # Tecplot 变量名称列表与数据列
        var_names = ['"X"', '"Y"']
        columns = [X_flat, Y_flat]
        
        # 依次提取 Q 数据的各个通道 (例如 u, v, p 等物理量)
        for c in range(num_channels):
            var_names.append(f'"Q{c+1}"')
            columns.append(Q_data[idx, c].flatten())
            
        # 将所有一维数组按列拼接，形成 N x (2 + channels) 的二维矩阵
        export_data = np.column_stack(columns)
        
        # 写入 Tecplot ASCII 格式
        with open(out_filename, 'w') as f:
            f.write(f'TITLE = "Pipe Flow Sample {idx}"\n')
            f.write(f'VARIABLES = {", ".join(var_names)}\n')
            # 定义网格结构: 结构化网格，POINT 模式按点逐行记录
            f.write(f'ZONE I={ny}, J={nx}, DATAPACKING=POINT\n')
            
            # 高效写入大量数值数据
            np.savetxt(f, export_data, fmt='%.6e', delimiter='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pipe .npy data to Tecplot .dat")
    parser.add_argument('--data_dir', type=str, default='./data/fno/pipe/', help='存放npy数据的目录')
    parser.add_argument('--out_dir', type=str, default='./tecplot_data/', help='Tecplot文件的输出目录')
    parser.add_argument('--samples', type=int, nargs='+', default=[0, 10, 100], help='需要导出的样本索引编号列表，用空格隔开')
    
    args = parser.parse_args()
    convert_to_tecplot(args.data_dir, args.out_dir, args.samples)