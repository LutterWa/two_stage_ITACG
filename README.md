# 三维两阶段时间角度计算制导律

本工程实现了基于速度不可控模型的混合tgo预测方法和基于牛顿法的ITACG算法

- [GetF.m](./GetF.m): 计算在特定大气和气动系数下，飞行器受到的升力、侧向力和阻力

- [dery.m](./dery.m): 飞行器的三维动力学方程

- [rk4.m](./rk4.m): 四阶龙格库塔法

- [parsave.m](./parsave.m): 在parfor中按仿真初值存储数据

- [Main.m](./Main.m): 两阶段样本采集函数

- [preprocess.py](./preprocess.py): 两阶段样本预处理

- [dnn.py](./dnn.py): 两阶段交班点速度和到达时间预测器

- [vehicle.m](./vehicle.py)：变速模型的tgo预测和ITACG