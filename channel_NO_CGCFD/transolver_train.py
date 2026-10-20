import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
import os
import sys

sys.path.append('/home/sqj/LeetNeuralOperator/Transolver/PDE-Solving-StandardBenchmark')

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer

parser = argparse.ArgumentParser('Training Transolver')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_Structured_Mesh_2D')
parser.add_argument('--n-hidden', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=3)
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0')
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

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main():
    inputX = torch.tensor(np.load(f"{args.data_path}/Channel_X.npy"), dtype=torch.float)
    inputY = torch.tensor(np.load(f"{args.data_path}/Channel_Y.npy"), dtype=torch.float)
    input = torch.stack([inputX, inputY], dim=-1)

    output = torch.tensor(np.load(f"{args.data_path}/Channel_Q.npy"), dtype=torch.float)
    output = output.permute(0, 2, 3, 1)

    params = torch.tensor(np.load(f"{args.data_path}/Channel_Params.npy"), dtype=torch.float)
    params = params.unsqueeze(1).unsqueeze(1).repeat(1, 200, 30, 1)

    # 动态样本切割逻辑
    N = input.shape[0]
    if N == 1:
        ntrain, ntest = 1, 1
    else:
        ntrain = int(0.8 * N)
        ntest = N - ntrain

    r1, r2 = args.downsamplex, args.downsampley
    s1, s2 = int(((200 - 1) / r1) + 1), int(((30 - 1) / r2) + 1)

    x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1, 2)
    x_test = input[-ntest:, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1, 2)
    
    fx_train = params[:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1, 2)
    fx_test = params[-ntest:, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1, 2)
    
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1, 3)
    y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1, 3)

    # 1. 先实例化
    x_normalizer = UnitTransformer(x_train)
    fx_normalizer = UnitTransformer(fx_train)
    y_normalizer = UnitTransformer(y_train)

    # 2. 调用 encode
    x_train, x_test = x_normalizer.encode(x_train), x_normalizer.encode(x_test)
    fx_train, fx_test = fx_normalizer.encode(fx_train), fx_normalizer.encode(fx_test)
    y_train = y_normalizer.encode(y_train)

    # 3. 最后再移动到 GPU 上
    x_normalizer.cuda()
    fx_normalizer.cuda()
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, fx_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, fx_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = get_model(args).Model(space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden,
                                  dropout=args.dropout, n_head=args.n_heads, Time_Input=False,
                                  mlp_ratio=args.mlp_ratio, fun_dim=2, out_dim=3,
                                  slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos,
                                  H=s1, W=s2).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    for ep in range(args.epochs):
        model.train()
        train_loss = 0
        for pos, fx, y in train_loader:
            x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x, fx)
            out = y_normalizer.decode(out)
            y_decoded = y_normalizer.decode(y)

            loss = myloss(out, y_decoded)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()

        if ep % 10 == 0:
            print("Epoch {} Train loss : {:.5f}".format(ep, train_loss / ntrain))

    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('./checkpoints', args.save_name + '.pt'))

if __name__ == "__main__":
    main()