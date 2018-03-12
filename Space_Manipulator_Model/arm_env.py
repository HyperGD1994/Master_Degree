import numpy as np
import math
from DynamicModel import DynamicModel

pi = math.pi

q = np.zeros((3, 1))
qd = np.zeros((3, 1))
qdd = np.zeros((3, 1))

v0 = np.array([[3, 2, 1]]).T
w0 = np.array([[1, 2, 3]]).T
vd0 = np.array([[0, 0, 0]]).T
wd0 = np.array([[0, 0, 0]]).T

R0 = np.array([[0, 0, 0]]).T
Q0 = np.array([[0, 0, 0]]).T
A0 = np.eye(3)

Fe = np.zeros((3, 6))
Te = np.zeros((3, 6))

F0 = np.array([[0, 0, 0]]).T
T0 = np.array([[0, 0, 0]]).T

tau = np.zeros(3)

model = DynamicModel()
model.forward_dynamics_RungeKutta(R0, A0, v0, w0, q, qd, F0, T0, Fe, Te, tau)

class ArmEnv(object):
    def __init__(self):
        pass

    def step(self, action):
        pass

    def reward(self):
        pass
