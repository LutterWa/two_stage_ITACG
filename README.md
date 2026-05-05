# 三维两阶段时间角度计算制导律

本工程实现了基于速度不可控模型的混合tgo预测方法和基于牛顿法的ITACG算法

## 运行环境
**硬件**
- Intel Xeon Gold 6248
- Nvidia Geforce RTX 3090

**软件**
- Octave 10.1.0
- Python 3.12.4
- keras 3.9.2
- Pytorch 2.5.1
- CUDA 12.5
- numpy 1.26.4
- scipy 1.14.0
- scikit-learn 1.5.2
- tensorboard 2.19.0
- matplotlib 3.9.1

## 文件说明
**Octave**
- [Main.m](./Main.m): 两阶段样本采集函数
- [latin.m](./latin.m)：拉丁超立方采样函数
- [rk4.m](./rk4.m): 四阶龙格库塔函数
- [dery.m](./dery.m): 飞行器的三维动力学方程函数
- [GetF.m](./GetF.m): 计算在特定大气和气动系数下，飞行器受到的升力、侧向力和阻力
- [parsave.m](./parsave.m): 在parfor中按仿真初值存储数据

**Python**
- [preprocess.py](./preprocess.py): 两阶段样本预处理
- [dnn.py](./dnn.py): 两阶段交班点速度和到达时间预测器
- [vehicle.py](./vehicle.py)：变速模型的基础模型
- [itacg.py](./itacg.py)：变速模型的ITACG实现和蒙特卡罗仿真
---
