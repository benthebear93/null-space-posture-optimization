import os
import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import sympy as sym

sym.init_printing()

from pathlib import Path
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=200)

from spatialmath import *
import dill


def posture_read():
    df = pd.read_excel(
        "../data/random_curve_pos.xlsx", header=None, names=None, index_col=None
    )
    testCount = df.shape[0]

    set_pos_val = []
    pos_val = []
    for i in range(0, testCount):
        for j in range(0, 6):
            a = df.iloc[i][j]
            pos_val.append(a)
        set_pos_val.append(pos_val)
        pos_val = []

    return set_pos_val


class OptimalIK:
    def __init__(self, time_step, accuracy, max_iteration):
        self.root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.time_step = time_step
        self.accuracy = accuracy
        self.max_iteration = max_iteration
        self.c = np.array([0.01, 0.01, 0.01, 0.01, 0.01])  # Tikhonov # 0.1
        self.F = np.array([-6.0, -6.0, 40.0, 0.0, 0.0, 0.0])
        self.init_q = np.array(
            [0.1745, 0.1745, 0.1745, 0.1745, 0.1745, 0.1745, 0]
        )  # avoid singurality
        self.K_ = 1.0
        self.Kn_ = 0.001  # 0.001
        self.Ktheta = np.diag(np.array([1.7, 5.9, 1.8, 0.29, 0.93, 0.49]))

    def fk(self, q):
        """ Forward kinematic

        Args:
            q (numpy): joint value for 6 dof + extra joint

        Returns:
            TF: Homogeneous transform matrix
        """
        q = np.asarray(q, dtype=float)
        q1, q2, q3, q4, q5, q6, q7 = q
        dh_param1 = np.array([0, 0.05, -pi / 2])  # d a alpah
        dh_param2 = np.array([0, 0.425, 0])
        dh_param3 = np.array([0.0534, 0, pi / 2])
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
        R = TF[:3, :-1]

        return TF

    def Joint_limit_check(self, q):
        """ 
        Joint limit check

        Args:
            q (numpy): joint value for 6 dof + extra joint

        Returns:
            q (numpy): changed joint values

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
        """ 
        success check

        Args:
            error (list): x,y,z,r,p,y error

        Returns:
            True (bool)
        """
        accuracy = self.accuracy
        if (
            abs(error[0]) < accuracy
            and abs(error[1]) < accuracy
            and abs(error[2]) < accuracy
            and abs(error[3]) < accuracy
            and abs(error[4]) < accuracy
        ):
            return True

    def null_space_method(self, posCount, q0, p_goal):
        """
        Null space projection method

        Args:
            posCount (int): posture number
            q0 (numpy): initial joint value
            p_goal (numpy): goal position

        Returns:
            q_n0 (numpy): final joint value
            p (numpy): final position
            dxyz (numpy): end effector deviation
        """
        max_iteration = 500000
        Ktheta_inv = np.linalg.inv(self.Ktheta)

        goal_R = rotation_from_euler(np.deg2rad(p_goal[3:6]))
        print("Goal_R :", goal_R)
        q_n0 = q0

        p = self.fk(q_n0)[:3, -1]  # position
        R = self.fk(q_n0)[:3, :-1]  # Rotation matrix

        p_goal[3] = goal_R[2][1]  # roll
        p_goal[4] = goal_R[2][0]  # pitch
        p_goal[5] = goal_R[1][0]  # yaw
        p_goal = np.array([p_goal[0], p_goal[1], p_goal[2], p_goal[4], p_goal[5], 0])
        print(p_goal[3], p_goal[4], p_goal[5])
        p = np.array(
            [p[0], p[1], p[2], R[2][1], R[2][0], R[1][0]]
        )  # shape miss match (6,1) # x,y,z R31, R32
        t_dot = p_goal[:5] - p[:5]  # redundancy Rz remove (5,1)
        δt = self.time_step
        c = self.c

        i = 0

        J_func = dill.load(open("../param_save/J_func_simp", "rb"))
        H_func = dill.load(open("../param_save/H_func_simp", "rb"))

        q_dot = np.array([0, 0, 0, 0, 0, 0, 0])
        while True:
            if self.is_success(t_dot):
                print("pos ", p)
                break
            q_n0 = self.Joint_limit_check(q_n0)
            q_n0 = q_n0 + (δt * q_dot)

            p = self.fk(q_n0)[:3, -1]
            R = self.fk(q_n0)[:3, :-1]  # Rotation matrix
            p = np.array(
                [p[0], p[1], p[2], R[2][1], R[2][0], R[1][0]]
            )  # shape (5,1) = [x, y, z R32, R31, R21]

            T = find_T(R)
            invT = np.linalg.inv(T)

            J = J_func(q_n0)
            J_a = (
                np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), invT]]) @ J
            )
            J_na = np.array([J_a[0], J_a[1], J_a[2], J_a[4], J_a[5]])
            # print(type(J_na), J_na.shape)
            # J_na = J_a[:5]
            # print(type(J_na), J_na.shape)
            p = np.array([p[0], p[1], p[2], p[4], p[5], 0])
            # print(p_goal,"\n", p)
            # print("")
            t_dot = p_goal[:5] - p[:5]  # redundancy Rz remove (5,1)

            gH = np.array(H_func(q_n0)[0])  # gradient of cost function
            gH = np.array([gH[0], gH[1], gH[2], 0, 0, 0, 0])
            psd_J = J_na.T @ np.linalg.inv(
                (J_na @ J_na.T + c.T @ np.eye(5))
            )  # + c.T@np.eye(5) (7,5)
            J_temp = J[:, :6] @ Ktheta_inv @ J[:, :6].T
            dxyz = J_temp @ self.F
            if i % 100 == 0:
                print("p_goal", p_goal, "\np", p)
                print("t_dof ", t_dot)
                print("  ")
                print(posCount, i, " t_dot: ", t_dot.T)

            q_dot = (
                self.K_ * psd_J @ t_dot - self.Kn_ * (np.eye(7) - (psd_J @ J_na)) @ gH
            )  # 6x5 5x1    - (6x6-6x5 5x6) 7x1

            i += 1
            # print("p_goal3", p_goal)
            if i > max_iteration:
                print("No convergence")
                break
        print("R : \n", R)
        print("Goal R: ", goal_R)
        rpy = euler_from_rotation(R)
        p[3] = np.rad2deg(rpy[0])  # yaw
        p[4] = np.rad2deg(rpy[1])  # pitch
        p[5] = np.rad2deg(rpy[2])  # roll
        return q_n0, p, dxyz

    def get_cnfs_null(self, method_fun, kwargs=dict()):
        set_pos = posture_read()
        J1, J2, J3, J4, J5, J6, index, pos, dx, dy, dz = ([] for i in range(11))

        for i in range(len(set_pos)):
            q, p, d_xyz = method_fun(i, self.init_q, np.array(set_pos[i]), **kwargs)
            print("pos #", i, " :", p, " ans : ", np.rad2deg(q))
            index.append(i)
            J1.append(q[0])
            J2.append(q[1])
            J3.append(q[2])
            J4.append(q[3])
            J5.append(q[4])
            J6.append(q[5])
            pos.append(np.around(np.array(set_pos[i]), decimals=4))
            dx.append(d_xyz[0])
            dy.append(d_xyz[1])
            dz.append(0.5 * (d_xyz[0] * d_xyz[0] + d_xyz[1] * d_xyz[1]))
        pos_record = pd.DataFrame(
            {
                "J1": np.rad2deg(J1),
                "J2": np.rad2deg(J2),
                "J3": np.rad2deg(J3),
                "J4": np.rad2deg(J4),
                "J5": np.rad2deg(J5),
                "J6": np.rad2deg(J6),
                "pos": pos,
                "dx": dx,
                "dy": dy,
                "dz": dz,
            },
            index=index,
        )
        pos_record.to_excel(
            "../data/opt_data_curved.xlsx",
            sheet_name="Sheet2",
            float_format="%.3f",
            header=True,
        )


if __name__ == "__main__":
    # Length of Links in meters
    pi = np.pi
    pi_sym = sym.pi
    PosPlane = OptimalIK(0.01, 0.0005, 50000)
    # q1 = np.array(
    #     [
    #         5.07278192,
    #         66.18588826,
    #         59.85810729,
    #         -80.32352617,
    #         62.41827459,
    #         77.62662842,
    #         0,
    #     ]
    # )
    # q1 = np.deg2rad(q1)
    # a = np.deg2rad([-122.4152, 42.8652, 179.8769])
    # print(a)
    # R = rotation_from_euler(a)
    # R = np.array(
    #     [
    #         [-0.7324, 0.5764, 0.3624],
    #         [0.0021, 0.5342, -0.8454],
    #         [-0.6808, -0.6184, -0.3925],
    #     ]
    # )
    # print("custom :", euler_from_rotation(R))
    # print("code: ", euler_from_matrix(R))
    # PosPlane.fk(q1)
    PosPlane.get_cnfs_null(method_fun=PosPlane.null_space_method)
