import numpy as np
import math
from DynamicModel import DynamicModel
from arm_config import ArmConfig
import matplotlib.pyplot as plt

pi = math.pi
model = DynamicModel()
config = ArmConfig()

class ArmEnv(object):

    def __init__(self):

        self.link_num = config.link_num
        n = self.link_num

        q = np.zeros((n, 1))
        self.joint = q


        v0 = np.array([[0, 0, 0]]).T
        w0 = np.array([[0, 0, 0]]).T
        vd0 = np.array([[0, 0, 0]]).T
        wd0 = np.array([[0, 0, 0]]).T

        R0 = np.array([[0, 0, 0]]).T
        Q0 = np.array([[0, 0, 0]]).T
        A0 = np.eye(3)

        self.base = np.hstack((R0, A0, v0, w0))

        AA = model.calc_coordinate_transform(A0, q)
        RR = model.calc_position(R0, AA, q)

        self.state_record = []
        self.state_record.append(RR[-1])

        end_effector_pos = RR[-1]
        base_pos = RR[0]

        self.s = np.hstack((end_effector_pos, base_pos))

        self.d_time = config.d_time
        self.get_point = False
        self.grab_counter = 0

        self.target_pos = np.array([0.5, 2, 0])

    def get_state(self):

        n = self.link_num

        q = self.joint

        R0 = self.base[:, 0].reshape(3, 1)
        A0 = self.base[:, 1:4]
        v0 = self.base[:, 4].reshape(3, 1)
        w0 = self.base[:, 5].reshape(3, 1)

        return q, R0, A0, v0, w0

    def step(self, action):

        n = self.link_num
        q, R0, A0, v0, w0= self.get_state()

        qd = action.reshape(2,1)


        AA = model.calc_coordinate_transform(A0, q)
        RR = model.calc_position(R0, AA, q)

        vv, ww = model.calc_velocity(AA, v0, w0, q, qd)

        R0 += vv[0] * self.d_time
        A0 = np.dot(model.aw_vector_rotation(w0), A0)
        v0 = vv[0]
        w0 = ww[0]

        q += qd * self.d_time
        q %= 2*pi

        self.base[:, 0] = R0.reshape(3)
        self.base[:, 1:4] = A0
        self.base[:, 4] = v0.reshape(3)
        self.base[:, 5] = w0.reshape(3)

        end_effector_pos = RR[-1]
        base_pos = RR[0]

        self.state_record.append(RR[-1])
        self.s = np.hstack((end_effector_pos, base_pos))

        return self.s


if __name__ == '__main__':
    arm_env = ArmEnv()

    s_dim = arm_env.s.shape[0]
    a_dim = arm_env.joint.shape[0]

    a_bound = [-pi, pi]

    for_plot = []
    for time in np.arange(0.0, 10, 0.1):

        state = arm_env.state_record
        a = np.random.normal(size=a_dim)*pi
        a = a.clip(-pi, pi)

        q, R0, A0, v0, w0= arm_env.get_state()
        print(time)

        for_plot.append(arm_env.step(a))
    fig = plt.figure()
    plt.plot(for_plot)
    plt.show()



