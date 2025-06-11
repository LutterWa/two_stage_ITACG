import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

l = 4
u = 128


def load_data(file, shuffle_flag=True):
    data_raw = loadmat(file)
    scaler = StandardScaler()
    x = scaler.fit_transform(data_raw["x"])
    y = np.dot(data_raw["y"], np.diag([1, 0.2]))

    if shuffle_flag:
        x, y = shuffle(x, y, random_state=1)

    print("training data load done!, y_mean:{} y_std:{}".format(np.mean(y, axis=0), np.std(y, axis=0)))
    dim = [x.shape[-1], y.shape[-1]]
    return x, y, dim


def init_network(l, u, dim):
    # 创建网络
    x = keras.layers.Input(shape=[dim[0]], name="input")
    xm = keras.layers.Dense(units=u, activation='gelu', name="hidden_0")(x)
    for i in range(l):
        xm = keras.layers.Dense(units=u, activation='gelu', name="hidden_{}".format(2 * i + 1))(xm) + xm
        xm = keras.layers.Dense(units=u, activation='gelu', name="hidden_{}".format(2 * i + 2))(xm)
    y = keras.layers.Dense(dim[1], name='output')(xm)
    model = keras.Model(inputs=x, outputs=y)
    return model


def train(path, h5file, lr=0.001):
    if not os.path.exists("model"):
        os.makedirs("model")
    x, y, dim = load_data(path)
    model = init_network(l, u, dim)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=lr))
    model.summary()

    def scheduler(epoch):
        return lr * 0.995 ** epoch

    rs = keras.callbacks.LearningRateScheduler(scheduler)
    tb = keras.callbacks.TensorBoard(log_dir='logs/dnn', write_images=True)
    model.fit(x, y, batch_size=50000, epochs=1000, validation_split=0.02, verbose=1, callbacks=[tb, rs])
    model.save(h5file)
    return model


def test(path, h5file):
    savepath = h5file.split("/")[1].split(".")[0]
    x, y, dim = load_data(path, shuffle_flag=False)
    model = keras.models.load_model(h5file)
    y_ = model.predict(x, batch_size=100000)
    savemat('../mats/test_{}.mat'.format(savepath), {"y": y, "y_": y_})


if __name__ == "__main__":
    file_path = "../mats/flight.mat"
    modelpath = "model/dnn.keras"
    train(file_path, modelpath, lr=0.001)
    test(file_path, modelpath)
