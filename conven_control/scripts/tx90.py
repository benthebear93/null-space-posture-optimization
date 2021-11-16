import numpy as np
import sympy as sym
from spatialmath import *


class Tx90:
    def __init__(self):
        self.fk(joint_params)

    def fk(self, joint_params):
        """
        Joint variables consisting of 7 parameters
        """
        pi = np.pi
        pi_sym = sym.pi
        joint_params = np.asarray(joint_params, dtype=float)
        q1, q2, q3, q4, q5, q6, q7 = joint_params
        dh_param1 = np.array([0, 0.05, -pi / 2])  # d a alpah
        dh_param2 = np.array([0, 0.425, 0])
        dh_param3 = np.array([0.05, 0, pi / 2])
        dh_param4 = np.array([0.425, 0, -pi / 2])
        dh_param5 = np.array([0, 0, pi / 2])
        dh_param6 = np.array([0.1, 0, 0])
        dh_param7 = np.array([0.1031, 0.17298, 0])

        T12 = Homgm(dh_param1, q1, offset=0)
        T23 = Homgm(dh_param2, q2, offset=-pi / 2)
        T34 = Homgm(dh_param3, q3, offset=pi / 2)
        T45 = Homgm(dh_param4, q4, offset=0)
        T56 = Homgm(dh_param5, q5, offset=0)
        T67 = Homgm(dh_param6, q6, offset=0)
        T7E = Homgm(dh_param7, 0, offset=0)

        TF = T12 @ T23 @ T34 @ T45 @ T56 @ T67 @ T7E
        p = TF[:3, -1]
        R = TF[:3, :-1]  # Rotation matrix
        # rpy = euler_from_rotation(R)
        # p[3] = rpy[0]
        # p[4] = rpy[1]
        # p[5] = rpy[2]
        return TF

    def Joint_limit_check(self, q):
        """ 
        Joint limit check
        """
        if q[0] > 3.14159:
            q[0] = -3.14159 + q[0]
        elif q[0] < -3.14159:
            q[0] = -q[0]

        if q[1] > 2.5744:
            q[1] = -2.57445 + q[1]
        elif q[1] < -2.2689:
            q[1] = -q[1]

        if q[2] > 2.5307:
            q[2] = -2.5307 + q[2]
        elif q[2] < -2.5307:
            q[2] = -q[2]

        if q[3] > 4.7124:
            q[3] = -4.7124 + q[3]
        elif q[3] < -4.7124:
            q[3] = -q[3]

        if q[4] > 2.4435:
            q[4] = -2.4435 + q[4]
        elif q[4] < -2.0071:
            q[4] = -q[4]

        if q[5] > 4.7124:
            q[5] = -4.7124 + q[5]
        elif q[5] < -4.7124:
            q[5] = -q[5]

        return q

    def is_success(self, error):

        accuracy = self.accuracy
        if (
            abs(error[0]) < accuracy
            and abs(error[1]) < accuracy
            and abs(error[2]) < accuracy
            and abs(error[3]) < 0.0005
            and abs(error[4]) < 0.0005
        ):
            return True
