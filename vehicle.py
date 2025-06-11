import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, tan, atan2, sqrt, exp
from random import uniform
from scipy.io import loadmat, savemat


class Target:  # 目标
    def __init__(self, position=None):
        if position is None:
            self.x, self.y, self.z = 0, 0, 0
        else:
            self.x, self.y, self.z = position
        self.v, self.gamma, self.psi, self.q, self.eta = 0, 0, 0, 0, 0
        self.am = None


class Vehicle:  # 飞行器
    def __init__(self, state=None, target=None):  # 构造函数
        self.pi = 3.141592653589793  # 圆周率
        self.RAD = 180 / self.pi  # 弧度转角度
        self.g = 9.80665  # 重力加速度
        self.rho = 1.225  # 大气密度
        self.clalpha, self.cd0, self.cdalpha = 49.056, 0.2604, 29.072  # 气动力系数

        if state is None:
            self.state = [0, -10000, 5000, 1000, 400, 0, 0, 0, 0, 84.6]
        else:
            self.state = state

        """自身状态"""
        self.S = 0.057  # 参考面积
        self.m = 84.6  # 重量kg
        self.k = (self.rho * self.S * self.cd0) / (2 * self.m)  # dv/dR动力学系数

        self.t = self.state[0]  # 时间
        self.x = self.state[1]  # x位置
        self.y = self.state[2]  # y位置
        self.z = self.state[3]  # z位置
        self.v = self.state[4]  # 速度
        self.gamma = self.state[5]  # 弹道倾角
        self.psi = self.state[6]  # 弹道偏角
        self.alpha = self.state[7]  # 攻角
        self.beta = self.state[8]  # 侧滑角

        self.qd = np.array([-90, 0]) / self.RAD  # 期望落角

        """相对状态"""
        if target is None:
            target = Target()

        self.target = target  # 目标
        self.R = 0.  # 弹目距离
        self.q = np.array([0., 0.])  # 视线角
        self.eta = np.array([0., 0.])  # 速度前置角
        self.Rdot = 0.  # 接近速度
        self.qdot = np.array([0., 0.])  # 视线角速率
        self.seeker()  # 更新相对状态
        self.am = np.array([0., 0.])  # 制导指令

        self.record = {"state": [], "R": [], "q": [], "qdot": [], "am": [], "eta": []}  # 全弹道历史信息

    def modify(self, state=None, los=False):  # 修改导弹初始状态
        self.record = {"state": [], "R": [], "q": [], "qdot": [], "am": [], "eta": []}  # 清空全弹道历史信息
        if state is None:
            state = [0,  # 时间
                     uniform(-15000, -10000),  # x位置
                     uniform(5000, 10000),  # y位置
                     uniform(-2000, 2000),  # z位置
                     uniform(400, 600),  # 速度
                     uniform(-30, 30) / self.RAD,  # 弹道倾角
                     uniform(-30, 30) / self.RAD,  # 弹道偏角
                     0,  # 攻角
                     0,  # 侧滑角
                     84.6]  # 重量kg
        self.state = np.array(state)
        self.qd = np.array([uniform(-90, -30), uniform(-30, 30)]) / self.RAD  # 期望落角
        self.refresh()  # 更新弹体状态
        self.seeker()  # 更新弹体状态
        if los is True:
            self.state[5] = self.q[0]
            self.state[6] = self.q[1]
            self.qd = self.q
            self.refresh()  # 更新弹体状态
            self.seeker()  # 更新弹目相对运动学关系

    def refresh(self):  # 更新系统状态
        # 限幅
        if self.state[5] > self.pi:
            self.state[5] -= 2 * self.pi
        elif self.state[5] < -self.pi:
            self.state[5] += 2 * self.pi

        if self.state[6] > self.pi:
            self.state[6] -= 2 * self.pi
        elif self.state[6] < -self.pi:
            self.state[6] += 2 * self.pi

        self.t, self.x, self.y, self.z, self.v, self.gamma, self.psi, self.alpha, self.beta, self.m = self.state

    def dynamic(self, state):
        t, x, y, z, v, gamma, psi, alpha, beta, m = state

        # 运动学
        dx = v * cos(gamma) * cos(psi)  # x北向
        dy = v * sin(gamma)  # y天向
        dz = -v * cos(gamma) * sin(psi)  # z东向

        # 动力学
        L, D, B = self.get_F(state, alpha, beta)  # 计算气动力
        g = self.g
        dv = -D / m - g * sin(gamma)  # 速度标量
        dgamma = (L - m * g * cos(gamma)) / (m * v)  # 弹道倾角
        dpsi = B / (m * v * cos(gamma))  # 弹道偏角

        # 其他方程
        dt = 1
        dalpha = 0
        dbeta = 0
        dm = 0

        dstate = np.array([dt, dx, dy, dz, dv, dgamma, dpsi, dalpha, dbeta, dm])
        return dstate

    def seeker(self):  # 计算弹目相对信息
        r = np.array([self.target.x, self.target.y, self.target.z]) - np.array([self.x, self.y, self.z])
        v = np.array([self.target.v * cos(self.target.gamma) * cos(self.target.psi),
                      self.target.v * sin(self.target.gamma),
                      -self.target.v * cos(self.target.gamma) * sin(self.target.psi)]) - \
            np.array([self.v * cos(self.gamma) * cos(self.psi),
                      self.v * sin(self.gamma),
                      -self.v * cos(self.gamma) * sin(self.psi)])

        self.R = np.linalg.norm(r, 2)  # 计算弹目距离
        self.q = np.array([atan2(r[1], np.linalg.norm([r[0], r[2]], 2)), -atan2(r[2], r[0])])  # 计算弹目视线角
        if self.target.am is None:
            self.target.q = self.q
        self.eta = np.array([self.gamma, self.psi]) - self.q  # 导弹速度前置角
        self.target.eta = np.array([self.target.gamma, self.target.psi]) - self.target.q  # 目标速度前置角
        self.Rdot = np.dot(r, v) / self.R
        self.qdot = np.array([self.target.v * sin(self.target.eta[0]) - self.v * sin(self.eta[0]),
                              (self.target.v * cos(self.target.eta[0]) * sin(self.target.eta[1]) +
                               self.v * cos(self.eta[0]) * sin(self.eta[1])) / cos(self.q[0])]) / self.R

    def guidance(self):  # 制导功能
        R, v, gamma, psi, qdot, q, qd, g, k = self.R, self.v, self.gamma, self.psi, self.qdot, self.q, self.qd, self.g, self.k
        m_mx = 10. * g

        mu = 4 * k * R
        N1 = (mu ** 2 * (1 + (mu - 1) * exp(mu))) / ((1 - exp(mu)) ** 2 - mu ** 2 * exp(mu))
        N2 = 3 * (mu * (mu + 2 + (mu - 2) * exp(mu))) / ((1 - exp(mu)) ** 2 - mu ** 2 * exp(mu))

        am = [np.clip(v ** 2 / R * (N1 * (q[0] - gamma) + N2 * (q[0] - qd[0])) + cos(gamma) * g, -m_mx, m_mx),
              np.clip(v ** 2 * cos(gamma) / R * (N1 * (q[1] - psi) + N2 * (q[1] - qd[1])), -m_mx, m_mx)]
        self.am = np.array(am)

    def control(self, h):
        alpha_bound = np.array([-15., 15.]) / self.RAD  # 攻角范围
        beta_bound = np.array([-15., 15.]) / self.RAD  # 侧向角范围
        dalpha_bound = self.alpha + np.array([-60., 60.]) / self.RAD * h  # 攻角变化率范围deg/s
        dbeta_bound = self.beta + np.array([-60., 60.]) / self.RAD * h  # 侧向角变化率范围deg/s

        Q = 0.5 * self.rho * (self.v ** 2)  # 动压

        alpha = np.clip((self.m * self.am[0]) / (Q * self.S * self.clalpha), dalpha_bound[0], dalpha_bound[1])
        beta = np.clip((self.m * self.am[1]) / (Q * self.S * self.clalpha), dbeta_bound[0], dbeta_bound[1])

        self.state[7] = np.clip(alpha, alpha_bound[0], alpha_bound[1])
        self.state[8] = np.clip(beta, beta_bound[0], beta_bound[1])

    def rk4(self, h):  # 四阶龙格库塔
        k1 = h * self.dynamic(self.state)
        k2 = h * self.dynamic(self.state + 0.5 * k1)
        k3 = h * self.dynamic(self.state + 0.5 * k2)
        k4 = h * self.dynamic(self.state + k3)
        self.state = self.state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def get_F(self, state, alpha, beta):
        t, x, y, z, v, gamma, psi, _, _, m = state
        Q = 0.5 * self.rho * (self.v ** 2)  # 动压

        D = (self.cd0 + self.cdalpha * (alpha ** 2 + beta ** 2)) * Q * self.S  # 阻力
        L = self.clalpha * alpha * Q * self.S  # 升力
        B = self.clalpha * beta * Q * self.S  # 侧向力

        return L, D, B

    def step(self, h=0.001):  # 单步运行
        self.refresh()  # 更新系统状态

        if self.Rdot <= 0 or self.y > 0:  # 弹道终止条件
            self.seeker()  # 导引
            self.guidance()  # 制导
            self.control(h)  # 控制
            self.rk4(h)  # 积分

            self.record["state"].append(self.state)
            self.record["R"].append(self.R)
            self.record["q"].append(self.q)
            self.record["qdot"].append(self.qdot)
            self.record["am"].append(self.am)
            self.record["eta"].append(self.eta)

            return False
        else:
            return True

    def plot_data(self):  # 绘制曲线
        states = np.array(self.record["state"])
        R = np.array(self.record["R"])
        q = np.array(self.record["q"])
        qdot = np.array(self.record["qdot"])
        ams = np.array(self.record["am"])
        eta = np.array(self.record["eta"])

        # plt.ion()

        fig = plt.figure(0)
        fig.clf()
        ax = fig.add_subplot(331, projection='3d')
        ax.plot(states[:, 3] / 1000, states[:, 1] / 1000, states[:, 2] / 1000)
        ax = fig.add_subplot(332)
        ax.plot(states[:, 1] / 1000, states[:, 2] / 1000)
        ax.set_title("xoy")
        ax = fig.add_subplot(333)
        ax.plot(states[:, 1] / 1000, states[:, 3] / 1000)
        ax.set_title("xoz")
        ax = fig.add_subplot(334)
        ax.plot(states[:, 0], states[:, 4])
        ax.set_title("v")
        ax = fig.add_subplot(335)
        ax.plot(states[:, 0], states[:, 5] * self.RAD)
        ax.set_title("gamma")
        ax = fig.add_subplot(336)
        ax.plot(states[:, 0], states[:, 6] * self.RAD)
        ax.set_title("psi")
        ax = fig.add_subplot(337)
        ax.plot(states[:, 0], states[:, 7] * self.RAD)
        ax.set_title("alpha")
        ax = fig.add_subplot(338)
        ax.plot(states[:, 0], states[:, 8] * self.RAD)
        ax.set_title("beta")
        ax = fig.add_subplot(339)
        ax.plot(states[:, 0], R)
        ax.set_title("R")

        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(231)
        ax.plot(states[:-1, 0], ams[:-1, 0])
        ax.set_title("am")
        ax = fig.add_subplot(234)
        ax.plot(states[:-1, 0], ams[:-1, 1])
        ax = fig.add_subplot(232)
        ax.plot(states[:-1, 0], qdot[:-1, 0] * self.RAD)
        ax.set_title("qdot")
        ax = fig.add_subplot(235)
        ax.plot(states[:-1, 0], qdot[:-1, 1] * self.RAD)
        ax = fig.add_subplot(233)
        ax.plot(states[:-1, 0], q[:-1, 0] * self.RAD)
        ax.set_title("q")
        ax = fig.add_subplot(236)
        ax.plot(states[:-1, 0], q[:-1, 1] * self.RAD)

        # plt.pause(0.1)
        plt.show()


def test_vehicle():
    vehicle = Vehicle()
    e = 0
    result = {"v_real": np.array([]), "v_pred": np.array([]), "tgo_real": np.array([]), "tgo_pred": np.array([])}
    for i in range(1000):
        vehicle.modify(los=True)  # state=[0., -10000., 5000., 1000., 400., 0., 0., 0., 0., 84.6],
        done = False
        h = 0.001

        v0 = vehicle.v
        x0 = -np.linalg.norm([vehicle.x, vehicle.z])
        a = (vehicle.rho * vehicle.S * -vehicle.cd0) / (2 * vehicle.m * cos(vehicle.gamma))
        bg = vehicle.g * tan(vehicle.gamma)
        bm = (2 * vehicle.m * vehicle.g ** 2 * cos(vehicle.gamma) * vehicle.cdalpha) / (
                vehicle.rho * v0 ** 2 * vehicle.S * vehicle.clalpha ** 2)
        b = bg + bm
        c = (v0 ** 2 - b / a) * exp(-2 * a * x0)

        q0 = vehicle.q[0]
        eta0 = -vehicle.eta[0]
        etaf = vehicle.q[0] - vehicle.qd[0]
        R0 = vehicle.R
        y0 = x0 * tan(q0)
        Y = lambda x: (-(eta0 + etaf) / R0 ** 2 * ((x - x0) / cos(q0)) ** 3 +
                       (2 * eta0 + etaf) / R0 * ((x - x0) / cos(q0)) ** 2 +
                       -eta0 * ((x - x0) / cos(q0))) / cos(q0) + x * tan(q0)

        V = lambda x: max(sqrt(max(c * exp(2 * a * x) + b / a, 0)), 1)

        # V = lambda x: max(sqrt(max(c * exp(2 * a * x) + b / a, 0)), 1) + sqrt(
        #     (2 * (y0 - Y(x)) * vehicle.g + v0 ** 2)) - v0

        xl = [x0, x0 * 2 / 3, x0 * 1 / 3, 0]
        vx = [V(x) * cos(vehicle.gamma) for x in xl]
        A = np.array([[x ** (len(xl) - i - 1) for i in range(len(xl))] for x in xl])
        B = [1 / v for v in vx]
        k = np.dot(np.linalg.inv(A), B)
        Tgo = lambda x: -sum([k[len(xl) - i - 1] / (i + 1) * x ** (i + 1) for i in range(len(xl))])

        tgo = []
        v = []
        y = []
        while done is False:
            done = vehicle.step(h)
            xm = -np.linalg.norm([vehicle.x, vehicle.z])
            tgo.append(Tgo(xm))
            v.append(V(xm))
            y.append(Y(xm))

        states = np.array(vehicle.record["state"])
        e_max = max(np.array(tgo)[:-2] + states[:-1, 0] - vehicle.t)
        e_min = min(np.array(tgo)[:-2] + states[:-1, 0] - vehicle.t)

        if -e_min > e_max:
            e_max = e_min

        print("脱靶量={:.4f} 飞行时间={:.4f}, 落角误差={:.4f}, {:.4f}, 最大预测误差={:.4f}".format(
            vehicle.R, vehicle.t, (vehicle.gamma - vehicle.qd[0]) * vehicle.RAD,
                                  180 - abs(vehicle.q[1] - vehicle.qd[1]) * vehicle.RAD, e_max))
        # vehicle.plot_data()
        e += e_max

        plt.ion()
        plt.clf()
        # # tgo
        # plt.plot(states[:, 0], np.array(tgo)[:-1], linestyle='--')
        # plt.plot(states[:, 0], vehicle.t - states[:, 0])
        # # v
        plt.plot(states[:, 0], np.array(v)[:-1], linestyle='--')
        plt.plot(states[:, 0], states[:, 4])
        # y
        # qs = np.array(vehicle.record["q"])
        # y_ = np.array([-np.linalg.norm([states[i, 1], states[i, 3]]) * tan(qs[i, 0]) for i in range(qs.shape[0])])
        # plt.plot(states[:, 1], np.array(y)[:-1], linestyle='--')
        # plt.plot(states[:, 1], states[:, 2])
        plt.pause(0.1)

        result["v_pred"] = np.append(result["v_pred"], np.array(v)[:-1])
        result["v_real"] = np.append(result["v_real"], states[:, 4])
        result["tgo_pred"] = np.append(result["tgo_pred"], np.array(tgo)[:-1])
        result["tgo_real"] = np.append(result["tgo_real"], vehicle.t - states[:, 0])
    # savemat('mats/tgo_analytical_predict_monte.mat', result)
    print(e)


if __name__ == '__main__':
    test_vehicle()
