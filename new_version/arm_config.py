import numpy as np
import math


class ArmConfig(object):
    def __init__(self):
        self.d_time = 0.05
        self.gravity = 0

        # link number: n
        self.link_num = 2
        n = self.link_num

        # connect_lower : 1xn
        self.connect_lower = np.array([0, 1])  # base's lower connect is -1

        # connect_upper : nxn
        self.connect_upper = np.array([[-1, 1],
                                       [0, -1]])

        # connect_base : 1xn
        # connect_end  : 1xn
        S0 = np.array([1, 0])
        SE = np.array([0, 1])
        self.connect_end = SE
        self.connect_base = S0

        # inertia : (n+1) x (3x3)
        inertia = np.zeros((n+1, 3, 3))
        inertia[0] = np.array([[10, 0, 0],
                               [0, 10, 0],
                               [0, 0, 10]])

        inertia[1] = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0.1]])

        inertia[2] = np.array([[1, 0, 0],
                               [0, 0.1, 0],
                               [0, 0, 1]])

        self.inertia = inertia

        # mass : 1x(n+1)
        self.mass = np.array([1000, 10, 10])

        # q_from_Bi_to_i : nx3
        Qi = np.zeros((n+1, 3))
        pi = math.pi
        Qi[0] = np.array([0, 0, 0])
        Qi[1] = np.array([0, 0, 0])
        self.q_from_Bi_to_i = Qi

        # orientation_of_endpoint : (n+1)xn
        Qe = np.zeros((n + 1, 3))
        Qe[2] = np.array([0, 0, pi / 2])

        self.orientation_of_endpoint = Qe

        # link vector : (n+2)x(n+2)x3
        # from link i to link j
        # (from base link 0 to joint j
        # or from link i to end point)
        cc = np.zeros((n+2, n+2, 3))
        cc[1, 1] = np.array([0, -0.5, 0])
        cc[2, 2] = np.array([0, -0.5, 0])

        cc[1, 2] = np.array([0, 0.5, 0])

        cc[0, 1] = np.array([0, 1, 0])
        cc[2, 3] = np.array([0, 0.5, 0])

        self.link_vector = cc

        self.e_z = np.array([[0, 0, 1]]).T


