**`pinn-channelnel.py`**：使用物理信息神经网络 (PINN) 求解 Navier-Stokes 方程，生成包含空间坐标、物理参数（入口速度和运动粘度）以及对应的速度场和压力场数据的训练数据集，并导出为 `.npy` 格式文件。   

**`train_data_plot.py`**：读取生成的 `.npy` 格式通道流数据文件，通过指定样本索引来绘制对应的速度场 (U/V) 和压力场 (P) 云图，用于数据的可视化与验证。   

**`transolver_train_channel.py`**：加载生成的数据集并划分训练集与测试集，构建并训练 Transolver 神经算子模型，在训练完成后将模型权重文件（如 `channel_UniPDE.pt`）保存至 `checkpoints` 目录。   

**`infer_by_transolver_model_channel.py`**：加载已训练完毕的 Transolver 模型权重，在自定义的动态网格密度和物理参数（如特定的入口速度和运动粘度）下进行零样本 (Zero-Shot) 推理，并渲染输出流场预测图。
