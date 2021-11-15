import time
import numpy as np
import sympy as sym
import pandas as pd

sym.init_printing()
# from math import atan2, sqrt
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=200)
from spatialmath import *
import dill
import os

root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


def posture_read():
    # load_wb = load_workbook("C:/Users/UNIST/Desktop/stiffness_estimation/test_z.xlsx", data_only=True)
    df = pd.read_excel(
        "../data/flat_data_v2.xlsx", header=None, names=None, index_col=None
    )
    testCount = df.shape[0]

    print("number of test: ", (testCount - 1) / 2)
    overall_posval = []
    pos_val = []
    for i in range(0, testCount):
        for j in range(0, 6):
            a = df.iloc[i][j]
            pos_val.append(a)
        overall_posval.append(pos_val)
        pos_val = []
    return overall_posval


def is_success(error):
    accuracy = 0.001
    if (
        abs(error[0]) < 0.0005
        and abs(error[1]) < 0.0005
        and abs(error[2]) < 0.0005
        and abs(error[3]) < 0.001
        and abs(error[4]) < 0.001
        and abs(error[5]) < 0.001
    ):
        return True


def Joint_limit_check(q):
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


def FK(joint_params):
    """
    Joint variables consisting of 6 parameters
    """
    joint_params = np.asarray(joint_params, dtype=float)
    q1, q2, q3, q4, q5, q6, q7 = joint_params
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
    # print(p)
    return TF


def simple_pseudo(
    pos_num, q0, p_goal, time_step=0.1, max_iteration=500000, accuracy=0.001
):

    Ktheta = np.diag(np.array([1.7, 5.9, 1.8, 0.29, 0.93, 0.49]))
    Ktheta_inv = np.linalg.inv(Ktheta)
    F = np.array([-6.0, -6.0, 40.0, 0.0, 0.0, 0.0])
    goal_R = rotation_from_euler(np.deg2rad(p_goal[3:6]))

    q_n0 = q0

    p = FK(q_n0)[:3, -1]
    R = FK(q_n0)[:3, :-1]  # Rotation matrix

    p_goal[3] = goal_R[2][1] # roll
    p_goal[4] = goal_R[2][0] # pitch
    p_goal[5] = goal_R[0][0] # yaw 

    p = np.array(
        [p[0], p[1], p[2], R[2][1], R[2][0], R[0][0]]
    )  # shape miss match (6,1) # x,y,z R31, R32

    t_dot = p_goal[:6] - p
    δt = time_step
    i = 0

    start_time = time.time()
    #J_func = dill.load(open(root + '/scripts/J_func_simp', "rb"))
    J_func = dill.load(open('../param_save/J_func_simp', "rb"))

    print("start runnign\n")
    q_dot = np.array([0, 0, 0, 0, 0, 0, 0])
    while True:
        if is_success(t_dot):
            print(f"Accuracy of {accuracy} reached")
            break
        q_n0 = Joint_limit_check(q_n0)
        q_n0 = q_n0 + (δt * q_dot)

        p = FK(q_n0)[:3, -1]
        R = FK(q_n0)[:3, :-1]  # Rotation matrix

        p = np.array(
            [ p[0], p[1], p[2], R[2][1], R[2][0], R[0][0] ]
        )  # shape miss match (6,1) # x,y,z R31, R32

        T = find_T(R)
        invT = np.linalg.inv(T)

        J = J_func(q_n0)
        J_a = np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), invT]]) @ J
        J_inv = np.linalg.pinv(J_a)

        t_dot = p_goal[:6] - p
        # c = np.array([0.01,0.01,0.01,0.01,0.01,0.01])

        J_temp = J[:, :6] @ Ktheta_inv @ J[:, :6].T
        dxyz = J_temp @ F[:6]
        # psd_J = J_a.T@ np.linalg.inv((J_a@J_a.T))#  + c.T@np.eye(6)
        q_dot = J_inv @ t_dot
        i += 1
        if i % 100 == 0:
            print(pos_num, i, " t_dot: ", t_dot.T)
            # print("devi :", dxyz)
        if i > max_iteration:
            print("No convergence")
            break
    print("R : ", R)
    print("Goal R: ", goal_R)
    print("    ")
    rpy = euler_from_rotation(R)
    # print("ans quat : ",quaternion_matrix(R))
    # print("rpy angs : ", np.rad2deg(rpy))
    p[3] = np.rad2deg(rpy[0])  # pitch roll yaw
    p[4] = np.rad2deg(rpy[1])
    p[5] = np.rad2deg(rpy[2])
    print(p[3], p[4], p[5])
    return q_n0, p, dxyz


def get_cnfs_null(method_fun, q0, kwargs=dict()):
    # pos = np.array([0.650, -0.35, -0.278, 0, 1.57079, 0])
    # q, p, d_xyz = method_fun(0, q0, pos, **kwargs)
    # print(0, " pos", p, "\n ans : ", np.rad2deg(q))
    overallpos = posture_read()
    J1, J2, J3, J4, J5, J6, index, pos, dx, dy, dz = ([] for i in range(11))
    for i in range(len(overallpos)):
        q, p, d_xyz = method_fun(i, q0, np.array(overallpos[i]), **kwargs)
        print(i, " pos", p, "\n ans : ", np.rad2deg(q))
        print("  ")
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
        dz.append(d_xyz[2])
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
        "../data/teets.xlsx",
        sheet_name="Sheet2",
        float_format="%.3f",
        header=True,
    )


if __name__ == "__main__":
    # Length of Links in meters
    pi = np.pi
    pi_sym = sym.pi
    # q1 = np.array([0, 0, 90, 0, 0, 0, 0])
    # q1 = np.deg2rad(q1)
    # FK(q1)
    get_cnfs_null(method_fun=simple_pseudo, q0=np.deg2rad([10, 10, 10, 10, 10, 10, 0]))
