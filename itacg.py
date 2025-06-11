import keras
import numpy as np
from random import seed, uniform
from math import sin, cos, tan, sqrt, exp
from vehicle import Target, Vehicle
from scipy.io import savemat

mean = np.array(
    [-6657.11940527820, 6988.18354972679, -16.5024714054997, 196.391178827473, -0.274875294270398, -0.00159606585129713,
     -62.6878122353835, -0.0456698377692581, 4197.77870823977])
std = np.array(
    [4085.77300019679, 2375.80315217220, 2572.01114790482, 107.588660655523, 0.437226936901501, 0.549190523363442,
     16.7145667698595, 17.4635129871998, 2145.17152053676])


class Itacg(Vehicle):
    def __init__(self, state=None, target=None):  # 构造函数
        self.d = 0  # 伪目标距离
        self.R_threshold = 1  # 切换伪目标的距离阈值
        super().__init__(state, target)
        self.net = keras.models.load_model("model/dnn.keras")
        self.td = self.get_tgo()  # 期望飞行时间

    def set_d(self, d, qd=None):  # 设置伪目标和期望落角
        self.d = d  # 伪目标到终点距离
        if qd is not None:
            self.qd = qd  # 期望落角

        xtd = -d * cos(self.qd[0]) * cos(self.qd[1])  # 伪目标x位置
        ytd = -d * sin(self.qd[0])  # 伪目标y位置
        ztd = d * cos(self.qd[0]) * sin(self.qd[1])  # 伪目标z位置

        self.target = Target([xtd, ytd, ztd])  # 伪目标
        # print("tf={:.4f}".format(self.get_tgo()))

    def newton_iteration_solve_d(self, td, verbose=2):  # 弦截法
        n, dn_1, dn, en = 0, 0, self.R, 1e3
        en_1 = td - self.get_tgo(dn_1)
        while abs(en) > 1e-3 and abs(dn - dn_1) > 1e-6:
            en = td - self.get_tgo(dn)
            dn_next = dn - 0.9 * en / (en - en_1) * (dn - dn_1)
            en_1, dn_1, dn = en, dn, dn_next
            n += 1
        if verbose == 2:
            print("迭代次数={}, dn={:.4f}".format(n, dn))
        self.set_d(dn)

    # def newton_iteration_solve_d(self, td, verbose=2):  # 弦截法
    #     n, dn = 0, self.R / 2
    #     delta = 0.1
    #     en = td - self.get_tgo(dn)
    #     en_nable = (td - self.get_tgo(dn + delta) - en) / delta
    #     while abs(en) > 1e-3 and abs(en_nable) > 1e-6:
    #         dn = dn - en / en_nable
    #         en = td - self.get_tgo(dn)
    #         en_nable = (td - self.get_tgo(dn + delta) - en) / delta
    #         n += 1
    #     if verbose == 2:
    #         print("迭代次数={}, dn={:.4f}".format(n, dn))
    #     self.set_d(dn)

    def get_tgo(self, d=None):
        if d is None:
            d = self.d
        if np.linalg.norm([self.x, self.y, self.z]) - d > self.R_threshold:
            inputs = (np.concatenate([self.state[1:7], self.qd * self.RAD, [d]])[np.newaxis, :] - mean) / std
            outputs = self.net.predict(inputs, verbose=0)  # 神经网络单步预测
            t0, v0 = outputs[0, 0], outputs[0, 1] * 5  # 从当前状态出发，到达伪目标时的时间和速度

            xtd = -d * cos(self.qd[0]) * cos(self.qd[1])  # 伪目标x位置
            ztd = d * cos(self.qd[0]) * sin(self.qd[1])  # 伪目标z位置

            t1 = self.polynomial_tgo(v0, -np.linalg.norm([xtd, ztd]))  # 从伪目标到真目标的时间
        else:
            t0 = 0
            t1 = self.polynomial_tgo(self.v, -np.linalg.norm([self.x, self.z]))  # 从当前位置到真目标的时间
        return t0 + t1

    def polynomial_tgo(self, v0, x0):
        if x0 == 0:
            return 0
        # 1.解析速度预测公式
        a = (self.rho * self.S * -self.cd0) / (2 * self.m * cos(self.qd[0]))  # 零升阻力项系数
        b = self.g * tan(self.qd[0]) + (2 * self.m * self.g ** 2 * cos(self.qd[0]) * self.cdalpha) / (
                self.rho * v0 ** 2 * self.S * self.clalpha ** 2)  # 诱导阻力项系数
        c = (v0 ** 2 - b / a) * exp(-2 * a * x0)  # 初始状态常系数
        V = lambda x: max(sqrt(max(c * exp(2 * a * x) + b / a, 0)), 1)  # 速度预测函数

        # 2.多项式速度拟合公式
        xl = [x0, x0 * 2 / 3, x0 * 1 / 3, 0]  # 多项式预测点
        vx = [V(x) * cos(self.qd[0]) for x in xl]  # 计算各预测点的速度
        A = np.array([[x ** (len(xl) - i - 1) for i in range(len(xl))] for x in xl])  # Ax=B方程系数矩阵A
        B = [1 / v for v in vx]  # Ax=B方程结果向量A
        k = np.dot(np.linalg.inv(A), B)  # 解方程求得多项式系数
        Tgo = lambda x: -sum([k[len(xl) - i - 1] / (i + 1) * x ** (i + 1) for i in range(len(xl))])  # tgo预测函数
        return Tgo(x0)

    def seeker(self, d=None):
        if d is None:
            d = self.d
        if np.linalg.norm([self.x, self.y, self.z]) - d < self.R_threshold:  # 距离伪目标小于阈值时，切换目标
            self.target = Target()
        super().seeker()


def test_itacg(task):
    vehicle = Itacg()
    tds, ads = [], []
    if task == "td":
        tds = [55., 60., 65., 70., 75., 80.]
        ads = [[-80, 10.]]
    elif task == "ad":
        tds = [60.]
        ads = [[-70, 10.], [-70, 20.], [-80, 10.], [-80, 20.]]
    h = 0.001

    for td in tds:
        for ad in ads:
            vehicle.modify(state=[0., -10000., 5000., 1000., 400., 0., 0., 0., 0., 84.6])
            vehicle.set_d(0, np.array(ad) / vehicle.RAD)  # 设置期望落角
            vehicle.newton_iteration_solve_d(td)  # 根据飞行时间计算伪目标

            done = False
            t, n = 0, int(1 / h)
            tgo = []
            while done is False:
                done = vehicle.step(h)
                if t % n == 0:
                    # if np.linalg.norm([vehicle.x, vehicle.y, vehicle.z]) - vehicle.d < vehicle.R_threshold:
                    #     vehicle.newton_iteration_solve_d(td - vehicle.t-h)  # 根据飞行时间计算伪目标
                    tgo.append(vehicle.get_tgo())
                else:
                    tgo.append(tgo[-1] - h)
                t += 1
            print("脱靶量={:.4f} 飞行时间={:.4f}, 落角误差={:.4f}, {:.4f}, 时间误差={:.4f}".format(
                vehicle.R, vehicle.t, (vehicle.q[0] + vehicle.qd[0]) * vehicle.RAD,
                                      180 - abs(vehicle.q[1] - vehicle.qd[1]) * vehicle.RAD, td - vehicle.t))
            savemat('mats/sim_td_{:d}_ad_{:d}_{:d}.mat'.format(
                int(td), -int(vehicle.qd[0] * vehicle.RAD), int(vehicle.qd[1] * vehicle.RAD)),
                dict(vehicle.record, **{"tgo": np.array(tgo)[:-1]}))
            vehicle.plot_data()


def monte_carlo():
    seed(318)  # 设置随机种子
    vehicle = Itacg()
    itr = 500
    result = {"state": [], "R": [], "q": [], "qdot": [], "eta": [], "am": [], "tgo": []}
    for i in range(itr):
        vehicle.modify()
        ad = [uniform(-85, -65), uniform(-25, 25)]  # 期望到达角度
        vehicle.set_d(0, np.array(ad) / vehicle.RAD)  # 设置期望落角
        td = vehicle.get_tgo(0) + uniform(1, 10)  # 期望飞行时间
        vehicle.newton_iteration_solve_d(td, verbose=0)  # 根据飞行时间计算伪目标

        done = False
        h = 0.01
        t, n = 0, int(1 / h)
        tgo = []
        while done is False:
            done = vehicle.step(h)
            if t % n == 0:
                tgo.append(vehicle.get_tgo())
            else:
                tgo.append(tgo[-1] - h)
            t += 1
        print("仿真次数={:d} 脱靶量={:.4f} 飞行时间={:.4f}, 落角误差={:.4f}, {:.4f}, 时间误差={:.4f}".format(
            i + 1, vehicle.R, vehicle.t, (vehicle.q[0] + vehicle.qd[0]) * vehicle.RAD,
            180 - abs(vehicle.q[1] - vehicle.qd[1]) * vehicle.RAD, td - vehicle.t))
        # 记录本次飞行结果
        result["state"].append(vehicle.record["state"])
        result["R"].append(vehicle.record["R"])
        result["q"].append(vehicle.record["q"])
        result["qdot"].append(vehicle.record["qdot"])
        result["eta"].append(vehicle.record["eta"])
        result["am"].append(vehicle.record["am"])
        result["tgo"].append(np.array(tgo)[:-1])

    savemat('mats/sim_monte_carlo.mat', result)


if __name__ == '__main__':
    test_itacg("td")
    test_itacg("ad")
    # monte_carlo()
