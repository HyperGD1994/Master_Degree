import numpy as np
import math
from DynamicModel import DynamicModel
import matplotlib.pyplot as plt

pi = math.pi
model = DynamicModel()


class ArmEnv(object):

    def __init__(self):

        q = np.zeros((3, 1))
        qd = np.zeros((3, 1))
        qdd = np.zeros((3, 1))

        self.joint = np.hstack((q, qd, qdd))

        v0 = np.array([[0, 0, 0]]).T
        w0 = np.array([[0, 0, 0]]).T
        vd0 = np.array([[0, 0, 0]]).T
        wd0 = np.array([[0, 0, 0]]).T

        R0 = np.array([[0, 0, 0]]).T
        Q0 = np.array([[0, 0, 0]]).T
        A0 = np.eye(3)

        self.base = np.hstack((R0, A0, v0, w0, vd0, wd0))

        AA = model.calc_coordinate_transform(A0, q)
        self.state = []
        RR = model.calc_position(R0, AA, q)
        self.state.append(RR[-1])

        self.d_time = model.d_time

    def step(self, action):

        q, qd, qdd, R0, A0, v0, w0, vd0, wd0 = self.get_state()

        tau = action

        Fe = np.zeros((3, 6))
        Te = np.zeros((3, 6))

        F0 = np.array([[0, 0, 0]]).T
        T0 = np.array([[0, 0, 0]]).T

        vd0, wd0, qdd = model.foward_dynamics( R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau)
        R0, A0, v0, w0, q, qd = model.forward_dynamics_RungeKutta( R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau)

        self.joint[:, 0] = q.reshape(3)
        self.joint[:, 1] = qd.reshape(3)
        self.joint[:, 2] = qdd.reshape(3)

        self.base[:, 0] = R0.reshape(3)
        self.base[:, 1:4] = A0
        self.base[:, 4] = v0.reshape(3)
        self.base[:, 5] = w0.reshape(3)
        self.base[:, 6] = vd0.reshape(3)
        self.base[:, 7] = wd0.reshape(3)

        AA = model.calc_coordinate_transform(A0, q)
        RR = model.calc_position(R0, AA, q)
        self.state.append(RR[-1])

    def reward(self):
        pass

    def reset(self):
        self.__init__()

    def get_state(self):
        q = self.joint[:, 0].reshape(3, 1)
        qd = self.joint[:, 1].reshape(3, 1)
        qdd = self.joint[:, 2].reshape(3, 1)

        R0 = self.base[:, 0].reshape(3, 1)
        A0 = self.base[:, 1:4]
        v0 = self.base[:, 4].reshape(3, 1)
        w0 = self.base[:, 5].reshape(3, 1)
        vd0 = self.base[:, 6].reshape(3, 1)
        wd0 = self.base[:, 7].reshape(3, 1)
        return q, qd, qdd, R0, A0, v0, w0, vd0, wd0


if __name__ == '__main__':
    arm_env = ArmEnv()

    desired_q = np.array([[0.3, 0.2, 0.1]]).T
    gain_spring = 10
    gain_dumper = 10

    d_time = arm_env.d_time
    q_ans = []
    for time in np.arange(0.0, 10, d_time):

        state = arm_env.state
        q, qd, qdd, R0, A0, v0, w0, vd0, wd0 = arm_env.get_state()
        q_ans.append(np.array(q.reshape(3)))
        print(time)
        tau = gain_spring * (desired_q - q) - gain_dumper * qd
        arm_env.step(tau)

    fig = plt.figure()
    plt.plot(q_ans)
    plt.show()




