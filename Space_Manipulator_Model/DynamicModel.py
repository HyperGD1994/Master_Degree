import numpy as np
import math


class DynamicModel:
    def __init__(self):
        self.connect_lower = np.array([0, 1, 2, 0, 4, 5]) # base's lower connect is -1

        self.connect_upper = np.array([[-1, 1, 0, 0, 0, 0],
                                       [0, -1, 1, 0, 0, 0],
                                       [0, 0, -1, 0, 0, 0],
                                       [0, 0, 0, -1, 1, 0],
                                       [0, 0, 0, 0, -1, 1],
                                       [0, 0, 0, 0, 0, -1]])

        S0 = np.array([1, 0, 0, 1, 0, 0])
        SE = np.array([0, 0, 1, 0, 0, 1])
        self.connect_end = SE
        self.connect_base = S0

        J_type = np.array(['R', 'R', 'P', 'R', 'R', 'R'])

        inertia = np.zeros((7, 3, 3))
        inertia[0] = np.array([[10, 0, 0],
                               [0, 10, 0],
                               [0, 0, 10]])

        inertia[1] = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0.1]])

        inertia[2] = np.array([[1, 0, 0],
                               [0, 0.1, 0],
                               [0, 0, 1]])

        inertia[3] = np.array([[1, 0, 0],
                               [0, 0.1, 0],
                               [0, 0, 1]])

        inertia[4] = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0.1]])

        inertia[5] = np.array([[1, 0, 0],
                               [0, 0.1, 0],
                               [0, 0, 1]])

        inertia[6] = np.array([[1, 0, 0],
                               [0, 0.1, 0],
                               [0, 0, 1]])
        self.inertia = inertia

        self.mass = np.array([100, 10, 10, 10, 10, 10, 10])
        Qi = np.zeros((6, 3))
        pi = math.pi
        Qi[0] = np.array([-pi / 2, 0, 0])
        Qi[1] = np.array([pi / 2, 0, 0])
        Qi[2] = np.array([0, 0, 0])
        Qi[3] = np.array([pi / 2, 0, 0])
        Qi[4] = np.array([pi / 2, 0, 0])
        Qi[5] = np.array([0, 0, 0])

        self.q_from_Bi_to_i = Qi

        Qe = np.zeros((6+1, 3))
        Qe[3] = np.array([0, 0, pi / 2])
        Qe[6] = np.array([0, 0, pi / 2])

        self.orientation_of_endpoint = Qe

        # link vector
        # from link i to link j
        # (from base link 0 to joint j
        # or from link i to end point)
        cc = np.zeros((8, 8, 3))
        cc[1, 1] = np.array([0, 0, -0.5])
        cc[2, 2] = np.array([0, 0, -0.5])
        cc[3, 3] = np.array([0, 0, -0.5])
        cc[4, 4] = np.array([0, 0, -0.5])
        cc[5, 5] = np.array([0, 0, -0.5])
        cc[6, 6] = np.array([0, 0, -0.5])

        cc[1, 2] = np.array([0, 0, 0.5])
        cc[2, 3] = np.array([0, 0, 0.5])
        cc[3, 4] = np.array([0, 0, 0.5])
        cc[4, 5] = np.array([0, 0, 0.5])
        cc[5, 6] = np.array([0, 0, 0.5])

        cc[0, 1] = np.array([0, 1, 0])
        cc[0, 4] = np.array([0, -1, 0])

        cc[3, 7] = np.array([0, 0.5, 0])
        cc[6, 7] = np.array([0, 0.5, 0])

        self.link_vector = cc

        self.link_num = J_type.shape[0]

        #z轴上的单位向量
        self.e_z = np.array([[0, 0, 1]]).T

        self.gravity = 0
        self.d_time = 0.01

    def foward_dynamics(self, R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau):
        """
        计算前向动力学
        :param R0:
        :param A0:
        :param v0:
        :param w0:
        :param q:
        :param qd:
        :param F0:
        :param T0:
        :param Fe:
        :param Te:
        :param tau:
        :return: vd0, wd0, qdd
        """

        n = self.link_num

        # calculation of coordinate transformation matrices
        AA = self.calc_coordinate_transform(A0, q)

        # calculation of position vectors
        RR = self.calc_position(R0, AA, q)

        # calculation of inertia matrices, HH
        HH = self.calc_inertia_matrices(RR, AA)

        # calculation of velocity dependent term, Force0
        # this is obtained by the RNE inverse dynamics computation with the accelerations and external forces zero
        qdd0 = np.zeros((n, 1))
        # 线、角加速度
        acc0 = np.zeros((3, 1))
        # 外力、外力矩
        fe0 = np.zeros((3, n))

        Force0 = self.recursive_Newton_Euler(RR, AA, v0, w0, acc0, acc0, q, qd, qdd0, fe0, fe0)

        # Force: forces on generalized coordinate
        # Force_ex : forces on the end points

        Force = np.zeros((6+n, 1))
        Force_ex = np.zeros((6+n, 1))

        Force[0:3] = F0
        Force[3:6] = T0

        Force[6:6+n] = tau.reshape(n, 1)

        # calculate external forces
        Fx = np.zeros((3, 1))
        Tx = np.zeros((3, 1))
        taux = np.zeros((n, 1))

        num_e = 0
        F_ex = np.zeros((n+6, n)) # 维数
        for i in range(n):
            if self.connect_end[i] == 1:
                joints = self.find_joint_connection(num_e)
                tmp = self.calc_jacobian_joints(RR, AA, joints)
                JJ_tx_i = tmp[0:3, :]
                JJ_rx_i = tmp[3:6, :]

                num_e += 1
                A_I_i = AA[i+1]
                Re0i = RR[i+1] - RR[0] + np.dot(A_I_i, self.link_vector[i+1, 7])

                Me_i_1 = np.hstack((np.eye(3), np.zeros((3,3))))
                Me_i_2 = np.hstack((self.tilde(Re0i), np.eye(3)))
                Me_i_3 = np.hstack((JJ_tx_i.T, JJ_rx_i.T))
                Me_i = np.vstack((Me_i_1, Me_i_2, Me_i_3))

                F_ex[:, i] = np.dot(Me_i, np.vstack((Fe[:, i].reshape(3, 1), Te[:, i].reshape(3, 1)))).reshape(n+6)

        Fx += F_ex[0:3, :].sum(1).reshape(3, 1)
        Tx += F_ex[3:6, :].sum(1).reshape(3, 1)
        taux += F_ex[6:6+n, :].sum(1).reshape(n, 1)

        Force_ex[0:3] = Fx
        Force_ex[3:6] = Tx
        Force_ex[6:6+n] = taux

        # calculation of the acceleration
        a_Force = Force - Force0 + Force_ex
        Acc = np.dot(np.linalg.inv(HH), a_Force)

        vd0 = Acc[0:3]
        wd0 = Acc[3:6]
        qdd = Acc[6:6+n]

        return vd0, wd0, qdd

    def calc_coordinate_transform(self, A0, q):
        """
        计算坐标转换矩阵
        :param A0: 基座姿态 3x3
        :param q: 关节角 nx1
        :return: link_num 个旋转矩阵，分别代表各关节的坐标转换，第0个为基座
        """

        link_num = q.shape[0]
        AA = np.zeros((link_num+1, 3, 3))
        AA[0] = A0
        for i in range(link_num):
            roll = self.q_from_Bi_to_i[i, 0]
            pitch = self.q_from_Bi_to_i[i, 1]
            yaw = self.q_from_Bi_to_i[i, 2] + q[i]
            tmp = self.rpy_to_direction_cosine(roll, pitch, yaw)

            AA[i+1] = np.dot(AA[self.connect_lower[i]], tmp.T)
        return AA

    def calc_position(self, R0, AA, q):
        """
        计算各连杆位置
        :param R0: 基座初始位置 3x1
        :param AA: 旋转矩阵，由calc_coordinate_transform函数计算
        :param q: 关节角，此处仅用来得到连杆数量 nx1
        :return: nx3矩阵，为每个连杆中心位置坐标（基准系下）
        """
        link_num = q.shape[0]
        RR = np.zeros((link_num+1, 3))
        RR[0] = R0.reshape(3)

        for i in range(link_num):
            RR[i+1] = RR[self.connect_lower[i]] + \
                      np.dot(AA[self.connect_lower[i]], self.link_vector[self.connect_lower[i], i+1]) - \
                      np.dot(AA[i+1], self.link_vector[i+1, i+1])

        return RR

    def calc_velocity(self, AA, v0, w0, q, qd):
        """
        计算各连杆速度
        :param AA: 旋转矩阵
        :param v0: 基座速度 3x1
        :param w0: 基座角速度 3x1
        :param q: 关节角 nx1
        :param qd: 关节角速度 nx1
        :return: 输出各连杆速度、角速度，均为 n+1x3x1矩阵， 输出均在惯性系下
        """
        n = self.link_num

        ww = np.zeros((n+1, 3, 1))
        ww[0] = w0

        vv = np.zeros((n+1, 3, 1))
        vv[0] = v0

        for i in range(n):
            A_I_BB = AA[self.connect_lower[i]]
            A_I_i = AA[i+1]

            ww[i+1] = ww[self.connect_lower[i]] + np.dot(A_I_i, self.e_z) * qd[i]
            # np的叉乘输入需要是行向量，进行了两次转置
            # 注意：叉乘时连杆矢量 linkvector 在哪个坐标系下表示，公式简写，计算需要进行坐标转换
            vv[i+1] = vv[self.connect_lower[i]] + \
                      np.cross(ww[self.connect_lower[i]].T, np.dot(A_I_BB, self.link_vector[self.connect_lower[i], i+1]).T).T - \
                      np.cross(ww[i+1].T, np.dot(A_I_i, self.link_vector[i+1, i+1]).T).T
        return vv, ww

    def calc_acceleration(self, AA, ww, vd0, wd0, q, qd, qdd):
        """
        计算各连杆加速度
        :param AA:
        :param ww:
        :param vd0:
        :param wd0:
        :param q:
        :param qd:
        :param qdd:
        :return: 输出各连杆 加速度、角加速度
        """
        n = self.link_num
        A_I_0 = AA[0]

        wd = np.zeros((n+1, 3, 1))
        wd[0] = wd0

        vd = np.zeros((n+1, 3, 1))
        vd[0] = vd0

        for i in range(n):
            A_I_BB = AA[self.connect_lower[i]]
            A_I_i = AA[i+1]

            B_i = self.connect_lower[i]

            wd[i+1] = wd[B_i] + \
                      np.cross(ww[i+1].T, np.dot(A_I_i, self.e_z).T * qd[i]).T + \
                      np.dot(A_I_i, self.e_z) * qdd[i]

            vd[i+1] = vd[B_i] + \
                      np.cross(wd[B_i].T, np.dot(A_I_BB, self.link_vector[B_i, i+1]).T).T + \
                      np.cross(ww[B_i].T, np.cross(ww[B_i].T, np.dot(A_I_BB, self.link_vector[B_i, i+1]).T)).T - \
                      np.cross(wd[i+1].T, np.dot(A_I_i, self.link_vector[i+1, i+1]).T).T - \
                      np.cross(ww[i+1].T, np.cross(ww[i+1].T, np.dot(A_I_i, self.link_vector[i+1, i+1]).T)).T
        return vd, wd

    def calc_inertia_matrices(self, RR, AA):
        """
        计算惯量矩阵 H (6+n)x(6+n)
        :param RR:
        :param AA:
        :return:
        """
        n = self.link_num
        Mass = self.mass.sum()

        # calculation of partial translational & rotational jacobian
        JJ_t = self.calc_translational_jacobian(RR, AA)
        JJ_r = self.calc_rotational_jacobians(AA)

        wE = Mass * np.eye(3)
        JJ_tg = np.zeros((3, n))
        HH_w = np.zeros((3, 3))
        HH_wq = np.zeros((3, n))
        HH_q = np.zeros((n, n))

        # position of gravity centroid Rg
        Rm = np.zeros((3))
        for i in range(n+1):
            Rm += self.mass[i] * RR[i]
        Rg = Rm / Mass

        wr0g = (Rg - RR[0]) * Mass

        for i in range(1, n+1):
            r0i = RR[i] - RR[0]
            A_I_i = AA[i]

            JJ_tg += self.mass[i] * JJ_t[i-1]

            HH_w += np.dot(np.dot(A_I_i, self.inertia[i]), A_I_i.T) \
                    + self.mass[i] * np.dot(self.tilde(r0i).T, self.tilde(r0i))
            HH_wq += np.dot(np.dot(np.dot(A_I_i, self.inertia[i]), A_I_i.T), JJ_r[i - 1]) \
                    + self.mass[i] * np.dot(self.tilde(r0i), JJ_t[i - 1])
            HH_q += np.dot(np.dot(JJ_r[i-1].T, np.dot(np.dot(A_I_i, self.inertia[i]), A_I_i.T)), JJ_r[i-1]) \
                    + self.mass[i] * np.dot(JJ_t[i-1].T, JJ_t[i-1])

        HH_w += np.dot(np.dot(AA[0], self.inertia[0]), AA[0].T)

        HH_1 = np.hstack((wE, self.tilde(wr0g).T, JJ_tg))
        HH_2 = np.hstack((self.tilde(wr0g), HH_w, HH_wq))
        HH_3 = np.hstack((JJ_tg.T, HH_wq.T, HH_q))

        HH = np.vstack((HH_1, HH_2, HH_3))
        return HH

    def calc_jacobian_joints(self, RR, AA, joints):
        n = len(joints)
        JJ_te = self.calc_translational_jacobians_joint(RR, AA, joints)
        JJ_re = self.calc_rotation_jacobians_joint(AA, joints)

        JJ = np.vstack((JJ_te, JJ_re))
        Jacobian = np.zeros((6, self.link_num))
        for i in range(n):
            Jacobian[:, joints[i]-1] = JJ[:, i]

        return Jacobian

    def recursive_Newton_Euler(self, RR, AA, v0, w0, vd0, wd0, q, qd, qdd, Fe, Te):
        """
        递归牛顿欧拉法，计算广义力
        :param RR:
        :param AA:
        :param v0:
        :param w0:
        :param vd0:
        :param wd0:
        :param q:
        :param qd:
        :param qdd:
        :param Fe:
        :param Te:
        :return: 广义力 (3+3+n)x1 向量
        """

        vv, ww = self.calc_velocity(AA, v0, w0, q, qd)
        vd, wd = self.calc_acceleration(AA, ww, vd0, wd0, q, qd, qdd)

        n = self.link_num

        FF0 = self.mass[0] * (vd[0] - self.gravity)
        TT0 = np.dot(np.dot(np.dot(AA[0], self.inertia[0]), AA[0].T), wd[0]) \
              + np.cross(ww[0].T, (np.dot(np.dot(np.dot(AA[0], self.inertia[0]), AA[0].T), ww[0])).T).T

        # calculation of inertia force & torque of each link
        FF = np.zeros((n+1, 3, 1))
        TT = np.zeros((n+1, 3, 1))

        for i in range(n+1):
            FF[i] = self.mass[i] * (vd[i] - self.gravity)

            TT[i] = np.dot(np.dot(np.dot(AA[i], self.inertia[i]), AA[i].T), wd[i]) \
                    + np.cross(ww[i].T, np.dot(np.dot(np.dot(AA[i], self.inertia[i]), AA[i].T), ww[i]).T).T

        # equilibrium of forces & torques on each link
        Fj = np.zeros((n, 3, 1))
        Tj = np.zeros((n, 3, 1))

        for i in range(n, 0, -1):

            F = np.zeros((3, 1))
            T = np.zeros((3, 1))

            for j in range(i+1, n+1):
                F += self.connect_upper[i-1, j-1] * Fj[j-1]
            Fj[i-1] = FF[i] + F - self.connect_end[i-1] * Fe[:, i-1].reshape(3, 1)

            for j in range(i+1, n+1):
                A_I_i = AA[i]
                T += self.connect_upper[i-1, j-1] * (np.cross(np.dot(A_I_i, (self.link_vector[i, j]-self.link_vector[i, i])).T, Fj[j-1].T).T + Tj[j-1])
            Tj[i-1] = TT[i] + T - np.cross(np.dot(AA[i], self.link_vector[i, i]).T, FF[i].T).T
            Tj[i-1] -= self.connect_end[i-1] * (np.cross(np.dot(AA[i], (self.link_vector[i, n+1] - self.link_vector[i, i])).T, Fe[:, i-1].reshape(1, 3)).T + Te[:, i-1].reshape(3, 1))

        # equilibrium on link 0
        F = np.zeros((3, 1))
        T = np.zeros((3, 1))

        for i in range(n):
            if self.connect_base[i]:
                F += self.connect_base[i] * Fj[i]
        FF0 += F

        for i in range(n):
            if self.connect_base[i]:
                T += self.connect_base[i] * (np.cross(np.dot(AA[0], self.link_vector[0,i+1]).T, Fj[i].T).T + Tj[i])
        TT0 += T

        # calculation of torques of each joint
        tau = np.zeros(n)
        for i in range(n):
            tau[i] = np.dot(Tj[i].T, np.dot(AA[i+1], self.e_z))

        Force = np.vstack((FF0, TT0, tau.reshape((-1, 1))))

        return Force

    def calc_translational_jacobian(self, RR, AA):
        """
        计算平动雅克比矩阵
        :param RR:
        :param AA:
        :return: link_num x (3 x link_num) 矩阵
        """
        link_num = self.link_num
        JJ_t = np.zeros((link_num, 3, link_num))
        for i in range(1, link_num+1):
            j = i
            while j > 0:
                A_I_j = AA[j]
                tmp1 = np.dot(A_I_j, self.e_z)
                tmp2 = RR[i] - RR[j] - np.dot(A_I_j, self.link_vector[j, j].T)
                tmp3 = np.cross(tmp1.reshape(3), tmp2)
                JJ_t[i-1][:, j-1] = tmp3
                j = self.connect_lower[j-1]
        return JJ_t

    def calc_translational_jacobians_joint(self, RR, AA, joints):
        """
        计算由 joints 给出特定关节的平动雅克比矩阵
        :param RR:
        :param AA:
        :param joints:
        :return: (nx3)' 矩阵
        """
        n = len(joints)
        POS, ORI = self.forward_kinematics_joints(RR, AA, joints)

        JJ_te = np.zeros((n, 3))
        for i in range(n):
            tmp = np.cross(np.dot(AA[joints[i]], self.e_z).T, (POS[n] - POS[i]).T).T
            JJ_te[i] = tmp.reshape(3)
        return JJ_te.T

    def calc_rotational_jacobians(self, AA):
        """
        计算转动雅克比矩阵
        :param AA:
        :return: ;link_num x (3 x link_num) 矩阵
        """
        link_num = self.link_num
        JJ_r = np.zeros((link_num, 3, link_num))
        for i in range(1, link_num + 1):
            j = i
            while j > 0:
                A_I_j = AA[j]
                tmp1 = np.dot(A_I_j, self.e_z).reshape(3)
                JJ_r[i-1][:, j-1] = tmp1
                j = self.connect_lower[j - 1]
        return JJ_r

    def calc_rotation_jacobians_joint(self, AA, joints):
        """
        计算由 joints 指出特定关节连接的转动雅克比矩阵
        :param AA:
        :param joints: 关节连接数据， 由 find joint connection 计算，为一个从小到大的列表
        :return: (nx3)' 矩阵
        """
        n = len(joints)
        JJ_re = np.zeros((n, 3))
        for i in range(n):
            A_I_i = AA[joints[i]]
            JJ_re[i] = np.dot(A_I_i, self.e_z).reshape(3)
        return JJ_re.T

    def forward_kinematics_joints(self, RR, AA, joints):
        """
        计算由 joints 给出的各关节 前向运动学
        :param RR:
        :param AA:
        :param joints:
        :return: 位置 (n+1)x(3x1) 前 n 项对应joints的n个关节，第n+1项为末端执行器； 姿态 (n+1)x(3x3)，同上
        """
        n = len(joints)
        k = joints[-1]
        POS_j = np.zeros((n+1, 3, 1))
        ORI_j = np.zeros((n+1, 3, 3))

        # each joints
        for i in range(n):
            ORI_tmp = AA[joints[i]]
            POS_tmp = RR[joints[i]] + np.dot(ORI_tmp, self.link_vector[joints[i], joints[i]])

            POS_j[i] = POS_tmp.reshape(3, 1)
            ORI_j[i] = ORI_tmp

        # effector
        roll = self.orientation_of_endpoint[k, 0]
        pitch = self.orientation_of_endpoint[k, 1]
        yaw = self.orientation_of_endpoint[k, 2]
        A_i_EE = self.rpy_to_direction_cosine(roll, pitch, yaw).T

        ORI_e = np.dot(AA[k], A_i_EE)
        POS_e = RR[k] + np.dot(AA[k], self.link_vector[k, 7])

        POS_j[n] = POS_e.reshape(3, 1)
        ORI_j[n] = ORI_e

        return POS_j, ORI_j

    def find_joint_connection(self, num_e):
        """
        计算 基座-末端 的连接顺序
        :param num_e: 第 num_e 条
        :return: 列表
        """
        n = self.link_num
        ie = np.zeros(self.connect_end.sum(), dtype=int)

        j = 0
        for i in range(n):
            if self.connect_end[i] == 1:
                ie[j] = i + 1
                j += 1

        joint_lower = self.connect_lower[ie[num_e]-1]
        connection = [ie[num_e]]

        while joint_lower != 0:
            connection.insert(0, joint_lower)
            joint_lower = self.connect_lower[joint_lower - 1]

        joint = connection

        return joint

    def rpy_to_direction_cosine(self, roll, pitch, yaw):

        Cx = np.array([[1, 0, 0],
                       [0, np.cos(roll), np.sin(roll)],
                       [0, -np.sin(roll), np.cos(roll)]])

        Cy = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                       [0, 1, 0],
                       [np.sin(pitch), 0, np.cos(pitch)]])

        Cz = np.array([[np.cos(yaw), np.sin(yaw), 0],
                      [-np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
        C = np.dot(Cz, np.dot(Cy, Cx))

        return C

    def tilde(self, A):
        """
        a scew-symmetric operator ~
        :param A: array(3)
        :return: 3x3 array
        """
        B = np.array([[0, -A[2], A[1]],
                      [A[2], 0, -A[0]],
                      [-A[1], A[0], 0]])
        return B

    def aw_vector_rotation(self, w0):
        dt = self.d_time
        w0 = w0.reshape(3)
        if np.linalg.norm(w0, 2)==0:
            E0 = np.eye(3)
        else:
            th = np.linalg.norm(w0, 2) * dt
            w = w0 / np.linalg.norm(w0, 2)

            E0 = np.array([[np.cos(th) + pow(w[0], 2) * (1 - np.cos(th)),
                            w[0] * w[1] * (1 - np.cos(th)) - w[2] * np.sin(th),
                            w[2] * w[0] * (1 - np.cos(th)) + w[1] * np.sin(th)],
                           [w[0] * w[1] * (1 - np.cos(th)) + w[2] * np.sin(th),
                            np.cos(th) + pow(w[1], 2) * (1 - np.cos(th)),
                            w[2] * w[1] * (1 - np.cos(th)) - w[0] * np.sin(th)],
                           [w[2] * w[0] * (1 - np.cos(th)) - w[1] * np.sin(th),
                            w[2] * w[1] * (1 - np.cos(th)) + w[0] * np.sin(th),
                            np.cos(th) + pow(w[2], 2) * (1 - np.cos(th))]])
        return E0


    def forward_dynamics_RungeKutta(self, R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau):
        """
        :param R0:
        :param A0:
        :param v0:
        :param w0:
        :param q:
        :param qd:
        :param F0:
        :param T0:
        :param Fe:
        :param Te:
        :param tau:
        :return:
        """
        dt = self.d_time

        tmp_vd0, tmp_wd0, tmp_qdd = self.foward_dynamics(R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau)
        K1_R0 = dt * v0
        K1_A0 = np.dot(self.aw_vector_rotation(w0), A0) - A0
        K1_q = dt * qd
        K1_v0 = dt * tmp_vd0
        K1_w0 = dt * tmp_wd0
        K1_qd = dt * tmp_qdd

        tmp_vd0, tmp_wd0, tmp_qdd = self.foward_dynamics(R0 + K1_R0 / 2, A0 + K1_A0 / 2, v0 + K1_v0 / 2, w0 + K1_w0 / 2,
                                                         q + K1_q / 2, qd + K1_qd / 2, F0, T0, Fe, Te, tau)
        K2_R0 = dt * (v0+K1_v0/2)
        K2_A0 = np.dot(self.aw_vector_rotation((w0+K1_w0/2)), A0) - A0
        K2_q = dt * (qd+K1_qd/2)
        K2_v0 = dt * tmp_vd0
        K2_w0 = dt * tmp_wd0
        K2_qd = dt * tmp_qdd

        tmp_vd0, tmp_wd0, tmp_qdd = self.foward_dynamics(R0 + K2_R0 / 2, A0 + K2_A0 / 2, v0 + K2_v0 / 2, w0 + K2_w0 / 2,
                                                         q + K2_q / 2, qd + K2_qd / 2, F0, T0, Fe, Te, tau)
        K3_R0 = dt * (v0 + K2_v0 / 2)
        K3_A0 = np.dot(self.aw_vector_rotation((w0 + K2_w0 / 2)), A0) - A0
        K3_q = dt * (qd + K2_qd / 2)
        K3_v0 = dt * tmp_vd0
        K3_w0 = dt * tmp_wd0
        K3_qd = dt * tmp_qdd

        tmp_vd0, tmp_wd0, tmp_qdd = self.foward_dynamics(R0 + K3_R0 / 2, A0 + K3_A0 / 2, v0 + K3_v0 / 2, w0 + K3_w0 / 2,
                                                         q + K3_q / 2, qd + K3_qd / 2, F0, T0, Fe, Te, tau)
        K4_R0 = dt * (v0 + K3_v0 )
        K4_A0 = np.dot(self.aw_vector_rotation((w0 + K3_w0 )), A0) - A0
        K4_q = dt * (qd + K3_qd )
        K4_v0 = dt * tmp_vd0
        K4_w0 = dt * tmp_wd0
        K4_qd = dt * tmp_qdd

        R0_next = R0 + (K1_R0 + 2 * K2_R0 + 2 * K3_R0 + K4_R0) / 6
        A0_next = A0 + (K1_A0 + 2 * K2_A0 + 2 * K3_A0 + K4_A0) / 6
        q_next = q + (K1_q + 2 * K2_q + 2 * K3_q + K4_q) / 6
        v0_next = v0 + (K1_v0 + 2 * K2_v0 + 2 * K3_v0 + K4_v0) / 6
        w0_next = w0 + (K1_w0 + 2 * K2_w0 + 2 * K3_w0 + K4_w0) / 6
        qd_next = qd + (K1_qd + 2 * K2_qd + 2 * K3_qd + K4_qd) / 6

        R0 = R0_next
        A0 = A0_next
        q = q_next
        v0 = v0_next
        w0 = w0_next
        qd = qd_next

        return R0, A0, v0, w0, q, qd


if __name__ == '__main__':
    pi = math.pi

    q = np.zeros((6,1))
    qd = np.zeros((6,1))
    qdd = np.zeros((6,1))

    v0 = np.array([[3, 2, 1]]).T
    w0 = np.array([[1, 2, 3]]).T
    vd0 = np.array([[0, 0, 0]]).T
    wd0 = np.array([[0, 0, 0]]).T

    R0 = np.array([[0, 0, 0]]).T
    Q0 = np.array([[0, 0, 0]]).T
    A0 = np.eye(3)

    Fe = np.zeros((3,6))
    Te = np.zeros((3,6))
    F0 = np.array([[0, 0, 0]]).T
    T0 = np.array([[0, 0, 0]]).T

    tau = np.zeros(6)

    model = DynamicModel()
    model.forward_dynamics_RungeKutta(R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau)
