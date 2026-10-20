import os
import sys

sys.path.append('/home/sqj/LeetNeuralOperator/Transolver/PDE-Solving-StandardBenchmark')

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer

parser = argparse.ArgumentParser('Training Transolver for Channel Flow')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
# parser.add_argument('--model', type=str, default='Transolver_2D')
parser.add_argument('--model', type=str, default='Transolver_Structured_Mesh_2D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='channel_UniPDE')
parser.add_argument('--data_path', type=str, default='./channel_data')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main():
    INPUT_X = args.data_path + '/Channel_X.npy'
    INPUT_Y = args.data_path + '/Channel_Y.npy'
    OUTPUT_Q = args.data_path + '/Channel_Q.npy'
    INPUT_PARAMS = args.data_path + '/Channel_Params.npy'

    # 数据集共100个样本，划分 80训练 / 20测试
    ntrain = 80
    ntest = 20
    N = 100

    r1 = args.downsamplex
    r2 = args.downsampley
    s1 = int(((200 - 1) / r1) + 1)
    s2 = int(((30 - 1) / r2) + 1)

    # 1. 加载并构建坐标输入 (Batch, 200, 30, 2)
    inputX = torch.tensor(np.load(INPUT_X), dtype=torch.float)
    inputY = torch.tensor(np.load(INPUT_Y), dtype=torch.float)
    input = torch.stack([inputX, inputY], dim=-1)

    # 2. 加载并构建预测目标 (Batch, 200, 30, 3)
    output = torch.tensor(np.load(OUTPUT_Q), dtype=torch.float)
    output = output.permute(0, 2, 3, 1)  # 从 (Batch, 3, X, Y) 转为 (Batch, X, Y, 3)

    # 3. 加载并构建物理参数特征 fx (Batch, 200, 30, 2)
    params = torch.tensor(np.load(INPUT_PARAMS), dtype=torch.float) # (Batch, 2)
    params = params.unsqueeze(1).unsqueeze(1).repeat(1, 200, 30, 1)

    print(f"Input Grid Shape: {input.shape}, Params Shape: {params.shape}, Output Shape: {output.shape}")

    # 下采样与序列化展平
    x_train = input[:N][:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1, 2)
    x_test = input[:N][-ntest:, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1, 2)
    
    fx_train = params[:N][:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1, 2)
    fx_test = params[:N][-ntest:, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1, 2)
    
    y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1, 3)
    y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1, 3)

    # 归一化处理
    x_normalizer = UnitTransformer(x_train)
    fx_normalizer = UnitTransformer(fx_train)
    y_normalizer = UnitTransformer(y_train)

    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    fx_train = fx_normalizer.encode(fx_train)
    fx_test = fx_normalizer.encode(fx_test)
    y_train = y_normalizer.encode(y_train)

    x_normalizer.cuda()
    fx_normalizer.cuda()
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, fx_train, y_train),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, fx_test, y_test),
        batch_size=args.batch_size, shuffle=False)

    print("Dataloading is over.")

    # 核心修改：fun_dim=2, out_dim=3
    model = get_model(args).Model(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=2,  
                                  out_dim=3,  
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  H=s1, W=s2).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    if eval:
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
        model.eval()
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        rel_err = 0.0
        showcase = 5
        id = 0

        with torch.no_grad():
            for pos, fx, y in test_loader:
                id += 1
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                out = model(x, fx) # 输入物理特征
                out = y_normalizer.decode(out)

                tl = myloss(out, y).item()
                rel_err += tl

                if id < showcase:
                    print(f"Saving plots for test sample {id}")
                    # 仅绘制 u 速度场 (channel 0) 用于可视化展示
                    grid_x = x[0, :, 0].reshape(s1, s2).detach().cpu().numpy()
                    grid_y = x[0, :, 1].reshape(s1, s2).detach().cpu().numpy()
                    pred_u = out[0, :, 0].reshape(s1, s2).detach().cpu().numpy()
                    gt_u = y[0, :, 0].reshape(s1, s2).detach().cpu().numpy()

                    plt.axis('off')
                    plt.pcolormesh(grid_x, grid_y, pred_u, shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.savefig(os.path.join('./results/' + save_name + '/', f"pred_u_{id}.pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.axis('off')
                    plt.pcolormesh(grid_x, grid_y, gt_u, shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.savefig(os.path.join('./results/' + save_name + '/', f"gt_u_{id}.pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    plt.axis('off')
                    plt.pcolormesh(grid_x, grid_y, pred_u - gt_u, shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.savefig(os.path.join('./results/' + save_name + '/', f"error_u_{id}.pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()

        rel_err /= ntest
        print("Test rel_err: {}".format(rel_err))
    else:
        for ep in range(args.epochs):
            model.train()
            train_loss = 0

            for pos, fx, y in train_loader:
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                optimizer.zero_grad()
                out = model(x, fx) # 将物理参数特征 fx 传递给模型

                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

                loss = myloss(out, y)
                loss.backward()

                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                train_loss += loss.item()
                scheduler.step()

            train_loss = train_loss / ntrain
            if ep % 10 == 0:
                print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))

            model.eval()
            rel_err = 0.0
            with torch.no_grad():
                for pos, fx, y in test_loader:
                    x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                    out = model(x, fx)
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest
            if ep % 10 == 0:
                print("Epoch {} Eval rel_err: {}".format(ep, rel_err))

            if ep % 100 == 0 or ep == args.epochs - 1:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

if __name__ == "__main__":
    main()