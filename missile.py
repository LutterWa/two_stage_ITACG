import keras
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, tan, atan2, sqrt, exp
from random import uniform
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(loadmat('mats/flight.mat')["x"])


class Target:  # 目标
    def __init__(self, position=None):
        if position is None:
            self.x, self.y, self.z = 0, 0, 0
        else:
            self.x, self.y, self.z = position
        self.v, self.gamma, self.psi, self.q, self.eta = 0, 0, 0, 0, 0
        self.am = None


class Missile:  # 常规制导武器
    def __init__(self, state=None, target=None):  # 构造函数
        self.pi = 3.141592653589793  # 圆周率
        self.RAD = 180 / self.pi  # 弧度转角度
        self.g = 9.80665  # 重力加速度
        self.rho = 1.225  # 大气密度
        self.clalpha, self.cd0, self.cdalpha = 49.056, 0.2604, 29.072  # 气动力系数

        if state is None:
            self.state = [0, -10000, 10000, 0, 400, 0, 0, 0, 0, 84.6]
        else:
            self.state = state

        """自身状态"""
        self.S = 0.057  # 参考面积
        self.m = 84.6  # 重量kg

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
        self.seeker()  # 更新弹目相对运动学关系
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
        L, D, B = self.get_F(state, alpha)  # 计算气动力
        g = self.g
        dv = -D / m - g * sin(gamma)  # 速度标量
        dgamma = (L - m * g * cos(gamma)) / (m * v)  # 弹道倾角
        dpsi = -B / (m * v * cos(gamma))  # 弹道偏角

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
        R, v, gamma, psi, qdot, q, qd, g = self.R, self.v, self.gamma, self.psi, self.qdot, self.q, self.qd, self.g
        tgo = R / v
        m_mx = 10. * g
        am = [np.clip(4 * v * qdot[0] + 2 * v * (q[0] - qd[0]) / tgo + cos(gamma) * g, -m_mx, m_mx),
              np.clip(4 * v * cos(gamma) * qdot[1] + 2 * v * cos(gamma) * (qd[1] - q[1]) / tgo, -m_mx, m_mx)]
        self.am = np.array(am)

    def control(self, h):
        alpha_bound = np.array([-30., 30.]) / self.RAD  # 攻角范围
        beta_bound = np.array([-30., 30.]) / self.RAD  # 侧向角范围
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

    def get_F(self, state, alpha):
        t, x, y, z, v, gamma, psi, _, beta, m = state
        Q = 0.5 * self.rho * (self.v ** 2)  # 动压

        cd = self.cd0 + self.cdalpha * alpha ** 2  # 阻力系数
        cl = self.clalpha * alpha  # 升力系数
        cb = self.clalpha * beta  # 侧向力系数

        L = cl * Q * self.S  # 升力
        D = cd * Q * self.S  # 阻力
        B = cb * Q * self.S  # 侧向力

        return L, D, B

    def step(self, h=0.001):  # 单步运行
        self.refresh()  # 更新系统状态

        if self.Rdot <= 0:  # 弹道终止条件
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


class Vehicle(Missile):
    def __init__(self, state=None, target=None):  # 构造函数
        self.d = 0  # 伪目标距离
        self.R_threshold = 20  # 切换伪目标的距离阈值
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

    def newton_iteration_solve_d(self, td):
        n, dn_1, dn, en = 0, 0, self.R, 1e3
        en_1 = td - self.get_tgo(dn_1)
        while abs(en) > 1e-3:
            en = td - self.get_tgo(dn)
            dn_next = dn - 0.8 * en / (en - en_1) * (dn - dn_1)
            en_1, dn_1, dn = en, dn, dn_next
            n += 1
        print("迭代次数={}, dn={:.4f}".format(n, dn))
        self.set_d(dn)

    def get_tgo(self, d=None):
        if d is None:
            d = self.d
        if np.linalg.norm([self.x, self.y, self.z]) - d > self.R_threshold:
            inputs = scaler.transform(np.concatenate([self.state[1:7], self.qd * self.RAD, [d]])[np.newaxis, :])
            outputs = self.net.predict(inputs, verbose=0)  # 神经网络单步预测
            t0, v0 = outputs[0, 0], outputs[0, 1] * 10  # 从当前状态出发，到达伪目标时的时间和速度

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
        b = self.g * tan(self.qd[0]) - (2 * self.m * self.g ** 2 * cos(self.qd[0]) * -self.cdalpha) / (
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


def test_miss():
    miss = Missile()
    e = 0
    for _ in range(100):
        miss.modify(los=False)  # state=[0., -10000., 10000., 0., 400., 0., 0., 0., 0., 100],
        done = False
        h = 0.001

        v0 = miss.v
        x0 = -np.linalg.norm([miss.x, miss.z])
        a = (miss.rho * miss.S * -miss.cd0) / (2 * miss.m * cos(miss.gamma))
        bg = miss.g * tan(miss.gamma)
        bm = (2 * miss.m * miss.g ** 2 * cos(miss.gamma) * -miss.cdalpha) / (
                miss.rho * v0 ** 2 * miss.S * miss.clalpha ** 2)
        b = bg - bm
        c = (v0 ** 2 - b / a) * exp(-2 * a * x0)

        V = lambda x: max(sqrt(max(c * exp(2 * a * x) + b / a, 0)), 1)

        xl = [x0, x0 * 2 / 3, x0 * 1 / 3, 0]
        vx = [V(x) * cos(miss.gamma) for x in xl]
        A = np.array([[x ** (len(xl) - i - 1) for i in range(len(xl))] for x in xl])
        B = [1 / v for v in vx]
        k = np.dot(np.linalg.inv(A), B)
        Tgo = lambda x: -sum([k[len(xl) - i - 1] / (i + 1) * x ** (i + 1) for i in range(len(xl))])

        tgo = []
        v = []
        while done is False:
            done = miss.step(h)

            xm = -np.linalg.norm([miss.x, miss.z])

            tgo.append(Tgo(xm))
            v.append(V(xm))

        states = np.array(miss.record["state"])
        e_max = max(np.array(tgo)[:-2] + states[:-1, 0] - miss.t)
        e_min = min(np.array(tgo)[:-2] + states[:-1, 0] - miss.t)

        if -e_min > e_max:
            e_max = e_min

        print("脱靶量={:.4f} 飞行时间={:.4f}, 落角误差={:.4f}, {:.4f}, 最大预测误差={:.4f}".format(
            miss.R, miss.t, (miss.gamma - miss.qd[0]) * miss.RAD, (miss.psi - miss.qd[1]) * miss.RAD, e_max))
        # miss.plot_data()
        e += e_max

        plt.ion()
        plt.clf()
        # tgo
        plt.plot(states[:, 0], np.array(tgo)[:-1])
        plt.plot(states[:, 0], miss.t - states[:, 0])
        # # v
        # plt.plot(states[:, 0], np.array(v)[:-1])
        # plt.plot(states[:, 0], states[:, 4])
        plt.pause(0.1)

    print(e)


def test_vehicle():
    miss = Vehicle()
    e = 0
    for td in [55., 60., 65., 70., 75., 80.]:
        miss.modify(state=[0., -10000., 10000., 1000., 400., -30. / miss.RAD, 0. / miss.RAD, 0., 0., 84.6])
        # miss.set_d(uniform(600, 6000), np.array([uniform(-85, -25), uniform(-30, 30)]) / miss.RAD)  # monte carlo
        miss.set_d(0, np.array([-80, 10.]) / miss.RAD)  # 设置期望落角
        miss.newton_iteration_solve_d(td)  # 根据飞行时间计算伪目标

        done = False
        h = 0.01

        tgo = []
        while done is False:
            done = miss.step(h)
            tgo.append(miss.get_tgo())

        states = np.array(miss.record["state"])
        e_max = max(np.array(tgo)[:-2] + states[:-1, 0] - miss.t)
        e_min = min(np.array(tgo)[:-2] + states[:-1, 0] - miss.t)

        if -e_min > e_max:
            e_max = e_min

        print("脱靶量={:.4f} 飞行时间={:.4f}, 落角误差={:.4f}, {:.4f}, 最大时间预测误差={:.4f}".format(
            miss.R, miss.t, (miss.q[0] + miss.qd[0]) * miss.RAD, 180 - abs(miss.q[1] - miss.qd[1]) * miss.RAD, e_max))
        # miss.plot_data()
        e += e_max ** 2
        plt.ion()
        plt.clf()
        # tgo
        plt.plot(states[:, 0], np.array(tgo)[:-1])
        plt.plot(states[:, 0], miss.t - states[:, 0])
        plt.pause(0.1)
        # plt.show()
    print(e)


if __name__ == '__main__':
    test_vehicle()
