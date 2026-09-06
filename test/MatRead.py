import scipy.io as scio
import argparse



def main():
    parser = argparse.ArgumentParser(description='Read a .mat file')
    parser.add_argument('--file', type=str, default='../data/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat', help='Path to the .mat file')
    args = parser.parse_args()

    data = scio.loadmat(args.file)
    # print(data.keys())
    # 1. 显示所有变量名
    print(f"\n📌 所有变量名 (共 {len(data)} 个):")
    for i, key in enumerate(data.keys(), 1):
        print(f"  {i}. {key}")

    # 2. 过滤掉MATLAB的元数据，只看实际数据
    actual_vars = [k for k in data.keys() if not k.startswith('__')]
    print(f"\n📊 实际数据变量 (共 {len(actual_vars)} 个):")
    # 3. 详细查看每个变量
    # for var_name in actual_vars:
    #     var_data = data[var_name]
    #     print(f"\n  🔹 {var_name}:")
    #     print(f"     类型: {type(var_data)}")
        
    # 获取变量
    u = data['u']
    a = data['a']
    t = data['t']
    print(f"   形状: {u.shape}")




















if __name__ == '__main__':
    main()