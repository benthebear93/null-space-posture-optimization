import os
import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import sympy as sym
from pathlib import Path

sym.init_printing()

import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=200)

from spatialmath import *
import dill
from tx90 import *


def posture_read():

    df = pd.read_excel(
        "../data/random_position_test_old.xlsx", header=None, names=None, index_col=None
    )
    num_test = df.shape[0]

    overall_posval = []
    pos_val = []
    for i in range(0, num_test):
        for j in range(0, 6):
            a = df.iloc[i][j]
            pos_val.append(a)
        overall_posval.append(pos_val)
        pos_val = []
    # print(overall_posval)
    return overall_posval


class OptimalIK(Tx90):
    def __init__(self, time_step, accuracy, max_iteration):
        self.K_ = 1.0
        self.Kn_ = 0.001  # 0.001

        self.root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.Ktheta = np.diag(np.array([1.7, 5.9, 1.8, 0.29, 0.93, 0.49]))
        self.time_step = time_step
        self.accuracy = accuracy
        self.max_iteration = max_iteration
        self.c = np.array([0.01, 0.01, 0.01, 0.01, 0.01])  # Tikhonov # 0.1
        self.F = np.array([-6.0, -6.0, 40.0, 0.0, 0.0, 0.0])
        self.init_q = np.array(
            [0.1745, 0.1745, 0.1745, 0.1745, 0.1745, 0.1745, 0]
        )  # avoid singurality
        # self.K         = np.array([[1.1], [1.1], [1.1], [0.1], [0.5], [1.1], [0]]) # avoid singurality

    def null_space_method(self, posCount, q0, p_goal):
        """
        Null space projection method
        """
        Ktheta_inv = np.linalg.inv(self.Ktheta)

        goal_R = rotation_from_euler(np.deg2rad(p_goal[3:6]))
        print("Goal R: ", goal_R)
        q_n0 = q0

        p = self.fk(q_n0)[:3, -1]  # position
        R = self.fk(q_n0)[:3, :-1]  # Rotation matrix

        p_goal[3] = goal_R[2][1]  # roll
        p_goal[4] = goal_R[2][0]  # pitch
        p_goal[5] = goal_R[1][0]  # yaw
        p_goal = np.array([p_goal[0], p_goal[1], p_goal[2], p_goal[4], p_goal[5], 0])

        p = np.array(
            [p[0], p[1], p[2], R[2][0], R[1][0], 0]
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
                print("Rotation: ", R)
                break
            q_n0 = self.Joint_limit_check(q_n0)
            q_n0 = q_n0 + (δt * q_dot)

            p = self.fk(q_n0)[:3, -1]
            R = self.fk(q_n0)[:3, :-1]  # Rotation matrix
            p = np.array(
                [p[0], p[1], p[2], R[2][0], R[1][0], 0]
            )  # shape (5,1) = [x, y, z R32, R31, R21]

            T = find_T(R)
            invT = np.linalg.inv(T)

            J = J_func(q_n0)
            J_a = (
                np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), invT]]) @ J
            )
            J_na = np.array([J_a[0], J_a[1], J_a[2], J_a[4], J_a[5]])

            t_dot = p_goal[:5] - p[:5]  # redundancy Rz remove (5,1)

            gH = np.array(H_func(q_n0)[0])  # gradient of cost function
            gH = np.array([gH[0], gH[1], gH[2], 0, 0, 0, 0])

            psd_J = J_na.T @ np.linalg.inv(
                (J_na @ J_na.T + c.T @ np.eye(5))
            )  # + c.T@np.eye(5) (7,5)
            J_temp = J[:, :6] @ Ktheta_inv @ J[:, :6].T
            dxyz = J_temp @ self.F
            if i % 100 == 0:
                print(posCount, i, " t_dot: ", t_dot.T)

            q_dot = (
                self.K_ * psd_J @ t_dot - self.Kn_ * (np.eye(7) - (psd_J @ J_na)) @ gH
            )  # 6x5 5x1    - (6x6-6x5 5x6) 7x1

            i += 1
            if i > self.max_iteration:
                print("No convergence")
                break
        rpy = euler_from_rotation(R)
        p[3] = np.rad2deg(rpy[0])  # yaw
        p[4] = np.rad2deg(rpy[1])  # pitch
        p[5] = np.rad2deg(rpy[2])  # roll
        return q_n0, p, dxyz

    def get_cnfs_null(self, method_fun, kwargs=dict()):
        overallpos = posture_read()
        J1, J2, J3, J4, J5, J6, index, pos, dx, dy, dz = ([] for i in range(11))

        for i in range(len(overallpos)):
            q, p, d_xyz = method_fun(i, self.init_q, np.array(overallpos[i]), **kwargs)
            print("pos\n", i, p, " ans : ", np.rad2deg(q))
            index.append(i)
            J1.append(q[0])
            J2.append(q[1])
            J3.append(q[2])
            J4.append(q[3])
            J5.append(q[4])
            J6.append(q[5])
            pos.append(np.around(np.array(overallpos[i]), decimals=4))
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
            self.root + "/data/opt_data_v2.xlsx",
            sheet_name="Sheet2",
            float_format="%.3f",
            header=True,
        )


if __name__ == "__main__":
    # Length of Links in meters
    pi = np.pi
    pi_sym = sym.pi
    PosPlane = OptimalIK(0.1, 0.001, 500000)
    # test = np.array([0.361652, 1.35713, 0.69029, 4.3405, 0.95651, 2.16569, 0]))
    # PosPlane.fk(test)
    # start = time.time()
    PosPlane.get_cnfs_null(method_fun=PosPlane.null_space_method)
