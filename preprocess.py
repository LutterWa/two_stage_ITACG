import os
import sys
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler


def progress_bar(i):
    print("\r", end="")
    print("{:.2f}%: ".format(i), "▋" * (int(i) // 2), end="")
    sys.stdout.flush()


def csv2mat(load_path):  # 回归器数据预处理，loadpath用二级目录，读取二级目录中的所有文件夹下的文件，整合后置于二级文件夹中。
    if load_path[-1] != '/':
        load_path += '/'

    files = os.listdir(load_path)  # 读取路径下的全部文件
    mat_files = []
    for f in files:  # 提取所有的csv
        if f.endswith(".mat"):
            mat_files.append(f)

    if not os.path.isdir("mats"):  # 创建目标文件夹
        os.mkdir("mats")

    # 解析csv文件
    l, h = 0, 100 / len(mat_files)
    for f in mat_files:
        l += h
        progress_bar(l)

        record = loadmat(load_path + f)["restate"]
        idx = record.shape[0]
        while record[idx - 1, -1] == 0:
            idx -= 1
        record = np.delete(record, slice(idx, record.shape[0]), 0)

        qgt = int(f[f.find("qgt") + 3:f.find("_qpt")])
        qpt = int(f[f.find("qpt") + 3:f.find("_d")])
        d = int(f[f.find("_d") + 2:f.find(".mat")])

        x = np.concatenate([record[:, 1:7],
                            np.ones([idx, 1]) * qgt,
                            np.ones([idx, 1]) * qpt,
                            np.ones([idx, 1]) * d], axis=1)

        tgo = record[-1, 0] - record[:, 0]
        y = np.concatenate([tgo[np.newaxis].T,
                            np.ones([idx, 1]) * record[-1, 4]], axis=1)

        try:
            x_cat = np.concatenate([x_cat, x])
            y_cat = np.concatenate([y_cat, y])
        except NameError:
            x_cat = x
            y_cat = y

    savemat('mats/flight.mat',
            {"x": x_cat,
             "y": y_cat[np.newaxis].T})


if __name__ == "__main__":
    csv2mat("data")
