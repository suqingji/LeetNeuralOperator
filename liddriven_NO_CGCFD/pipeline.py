import os
import argparse
import subprocess
import sys

def run_cmd(cmd):
    print(f"\n========== 正在执行: {' '.join(cmd)} ==========")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='自动化执行 PINN -> Transolver 流程')
    parser.add_argument('--num_samples', type=int, default=1, help='设置要生成的样本总量 (如 1 或 100)')
    parser.add_argument('--pinn_epochs', type=int, default=1000, help='PINN生成数据时的迭代次数')
    parser.add_argument('--transolver_epochs', type=int, default=50, help='Transolver训练的Epoch数')
    args = parser.parse_args()

    # 1. 运行 PINN 生成数据
    run_cmd([sys.executable, "pinn.py", "--num_samples", str(args.num_samples), "--epochs", str(args.pinn_epochs)])
    
    # 2. 绘制第0个样本以验证
    run_cmd([sys.executable, "plot.py", "--idx", "0"])
    
    # 3. 训练 Transolver
    run_cmd([sys.executable, "trainsolver_train.py", "--epochs", str(args.transolver_epochs)])
    
    # 4. 执行推理验证 (使用固定物理参数测试泛化性或拟合度)
    run_cmd([sys.executable, "infer.py", "--u_in", "5.0", "--nu", "0.05"])

    print("\n========== 整个流程顺利执行完毕！ ==========")

if __name__ == "__main__":
    main()