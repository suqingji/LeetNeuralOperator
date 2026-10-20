import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
import os
import sys
sys.path.append('/home/sqj/LeetNeuralOperator/Transolver/PDE-Solving-StandardBenchmark')
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from model_dict import get_model
from utils.normalizer import UnitTransformer

def main():
    parser = argparse.ArgumentParser('Transolver Zero-Shot Inference')
    parser.add_argument('--model_name', type=str, default='Transolver_Structured_Mesh_2D')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/channel_UniPDE.pt')
    parser.add_argument('--data_path', type=str, default='./channel_data')
    parser.add_argument('--nx', type=int, default=100)
    parser.add_argument('--ny', type=int, default=10)
    parser.add_argument('--u_in', type=float, default=5.0)
    parser.add_argument('--nu', type=float, default=0.05)
    parser.add_argument('--n-hidden', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--mlp_ratio', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--unified_pos', type=int, default=0)
    parser.add_argument('--ref', type=int, default=8)
    parser.add_argument('--slice_num', type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 动态匹配重建数据的形状
    full_data = np.load(f"{args.data_path}/Channel_X.npy")
    N = full_data.shape[0]
    ntrain = 1 if N == 1 else int(0.8 * N)

    inputX = torch.tensor(full_data[:ntrain], dtype=torch.float)
    inputY = torch.tensor(np.load(f"{args.data_path}/Channel_Y.npy")[:ntrain], dtype=torch.float)
    input_orig = torch.stack([inputX, inputY], dim=-1).reshape(ntrain, -1, 2)
    
    output_orig = torch.tensor(np.load(f"{args.data_path}/Channel_Q.npy")[:ntrain], dtype=torch.float)
    output_orig = output_orig.permute(0, 2, 3, 1).reshape(ntrain, -1, 3)
    
    params_orig = torch.tensor(np.load(f"{args.data_path}/Channel_Params.npy")[:ntrain], dtype=torch.float)
    params_orig = params_orig.unsqueeze(1).unsqueeze(1).repeat(1, 200, 30, 1).reshape(ntrain, -1, 2)

    x_normalizer = UnitTransformer(input_orig).to(device)
    fx_normalizer = UnitTransformer(params_orig).to(device)
    y_normalizer = UnitTransformer(output_orig).to(device)

    x_grid, y_grid = np.linspace(0, 10, args.nx), np.linspace(0, 1, args.ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    x_infer = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32).reshape(1, -1, 2).to(device)
    params_infer = torch.tensor([args.u_in, args.nu], dtype=torch.float32)
    fx_infer = params_infer.view(1, 1, 2).repeat(1, args.nx * args.ny, 1).to(device)

    x_infer_encoded = x_normalizer.encode(x_infer)
    fx_infer_encoded = fx_normalizer.encode(fx_infer)

    args.model = args.model_name
    model = get_model(args).Model(space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden,
                                  dropout=args.dropout, n_head=args.n_heads, Time_Input=False,
                                  mlp_ratio=args.mlp_ratio, fun_dim=2, out_dim=3,
                                  slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos,
                                  H=args.nx, W=args.ny).to(device)
    
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        out_encoded = model(x_infer_encoded, fx_infer_encoded)
        out = y_normalizer.decode(out_encoded)
    
    pred_u, pred_v, pred_p = out[0, :, 0].reshape(args.nx, args.ny).cpu().numpy(), out[0, :, 1].reshape(args.nx, args.ny).cpu().numpy(), out[0, :, 2].reshape(args.nx, args.ny).cpu().numpy()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    fig.suptitle(f"Transolver Inference | Grid: {args.nx}x{args.ny}, $U_{{in}}$={args.u_in}, $\\nu$={args.nu}", fontsize=14)

    axes[0].pcolormesh(X, Y, pred_u, shading='auto', cmap='jet')
    axes[1].pcolormesh(X, Y, pred_v, shading='auto', cmap='jet')
    axes[2].pcolormesh(X, Y, pred_p, shading='auto', cmap='jet')

    plt.tight_layout()
    plt.savefig(f"infer_U{args.u_in}_nu{args.nu}_grid{args.nx}x{args.ny}.png", dpi=150)

if __name__ == '__main__':
    main()