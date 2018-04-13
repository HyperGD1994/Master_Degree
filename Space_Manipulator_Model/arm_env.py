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
        qd = np.zeros((n, 1))
        qdd = np.zeros((n, 1))

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
        RR = model.calc_position(R0, AA, q)
        self.state_record = []
        self.state_record.append(RR[-1])

        end_effector_pos = RR[-1]
        base_pos = RR[0]

        self.s = np.hstack((end_effector_pos, base_pos))

        self.d_time = config.d_time

        # self.grab_buffer = 0.3
        self.get_point = False
        self.grab_counter = 0

        self.target_pos = np.array([0.5, 2, 0])
        self.best_distance = np.sqrt(np.sum(np.square(self.s[0:3] - self.target_pos)))

    def step(self, action, grab_buffer):

        # n = self.link_num
        # q, qd, qdd, R0, A0, v0, w0, vd0, wd0 = self.get_state()
        #
        # q += action.reshape(2,1) * model.d_time
        # q %= 2 * pi
        #
        # AA = model.calc_coordinate_transform(A0, q)
        # RR = model.calc_position(R0, AA, q)
        #
        # end_effector_pos = RR[-1]
        # base_pos = RR[0]
        #
        # self.state_record.append(RR[-1])
        # self.s = np.hstack((end_effector_pos, base_pos))
        # r = self.reward(grab_buffer)
        #
        # return self.s, r, self.get_point

        n = self.link_num
        q, qd, qdd, R0, A0, v0, w0, vd0, wd0 = self.get_state()

        tau = action

        Fe = np.zeros((3, 6))
        Te = np.zeros((3, 6))

        F0 = np.array([[0, 0, 0]]).T
        T0 = np.array([[0, 0, 0]]).T

        vd0, wd0, qdd = model.foward_dynamics( R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau)
        R0, A0, v0, w0, q, qd = model.forward_dynamics_RungeKutta( R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau)

        q %= (2 * pi)

        self.joint[:, 0] = q.reshape(n)
        self.joint[:, 1] = qd.reshape(n)
        self.joint[:, 2] = qdd.reshape(n)

        self.base[:, 0] = R0.reshape(3)
        self.base[:, 1:4] = A0
        self.base[:, 4] = v0.reshape(3)
        self.base[:, 5] = w0.reshape(3)
        self.base[:, 6] = vd0.reshape(3)
        self.base[:, 7] = wd0.reshape(3)

        AA = model.calc_coordinate_transform(A0, q)
        RR = model.calc_position(R0, AA, q)
        end_effector_pos = RR[-1]
        base_pos = RR[0]

        vel, _ = model.calc_velocity(AA, v0, w0, q, qd)
        end_v = vel[-1].reshape(3)
        self.state_record.append(RR[-1])
        self.s = np.hstack((end_effector_pos, base_pos, end_v))
        r = self.reward(grab_buffer)

        return self.s, r, self.get_point


    def reward(self, grab_buffer):
        t = 50

        # desired_pos = np.array([0, 3, 0.5, 0, 0, 0])
        desired_pos = self.target_pos

        distance = self.s[0:3] - desired_pos

        abs_distance = np.sqrt(np.sum(np.square(distance)))

        # r = -math.log(abs_distance)
        r = -abs_distance/10

        # if abs_distance > 3:
        #     self.get_point = True
        #     r = -100
        #     return r

        # if abs_distance < self.best_distance:
        #     self.best_distance = abs_distance
        #     r += 1

        if abs_distance <grab_buffer and (not self.get_point):
            r += 10
            self.grab_counter += 1
            if self.grab_counter > t:
                r += 100
                self.get_point = True

        elif abs_distance > grab_buffer:
            self.grab_counter = 0
            self.get_point = False


        return r

    def reset(self):
        self.__init__()
        return self.s

    def get_state(self):

        n = self.link_num

        q = self.joint[:, 0].reshape(n, 1)
        qd = self.joint[:, 1].reshape(n, 1)
        qdd = self.joint[:, 2].reshape(n, 1)

        R0 = self.base[:, 0].reshape(3, 1)
        A0 = self.base[:, 1:4]
        v0 = self.base[:, 4].reshape(3, 1)
        w0 = self.base[:, 5].reshape(3, 1)
        vd0 = self.base[:, 6].reshape(3, 1)
        wd0 = self.base[:, 7].reshape(3, 1)
        return q, qd, qdd, R0, A0, v0, w0, vd0, wd0


if __name__ == '__main__':
    arm_env = ArmEnv()

    s_dim = arm_env.s.shape[0]
    a_dim = arm_env.joint.shape[0]

    a_bound = [-10, 10]


    desired_q = np.array([[0.3, 0.2]]).T
    gain_spring = 10
    gain_dumper = 10

    d_time = arm_env.d_time
    q_ans = []
    for time in np.arange(0.0, 10, d_time):

        state = arm_env.state_record
        a = np.random.normal(size=a_dim)*10
        a = a.clip(-10, 10)

        q, qd, qdd, R0, A0, v0, w0, vd0, wd0 = arm_env.get_state()
        q_ans.append(np.array(q.reshape(2)))
        print(time)

        arm_env.step(a,grab_buffer=0.5)

    arm_env.reset()

    fig = plt.figure()
    plt.plot(state)
    plt.show()



