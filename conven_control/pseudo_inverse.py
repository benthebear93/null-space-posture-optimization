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

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

def posture_read():
    # load_wb = load_workbook("C:/Users/UNIST/Desktop/stiffness_estimation/test_z.xlsx", data_only=True)
    df = pd.read_excel('random_position.xlsx', header=None, names=None, index_col=None)
    num_test = df.shape[0]

    print("number of test: ",  (num_test-1)/2)
    overall_posval =[]
    pos_val = []
    for i in range(0, num_test):
        for j in range(0, 6):
            a = df.iloc[i][j]
            pos_val.append(a)
        overall_posval.append(pos_val)
        pos_val = []
    print(overall_posval)
    return overall_posval

root = os.getcwd()

def is_success(error):
    accuracy = 0.001
    if abs(error[0]) < accuracy and abs(error[1]) < accuracy and abs(error[2]) < accuracy and abs(error[3]) < 0.01 and abs(error[4]) < 0.01: 
        return True

def Joint_limit_check(q):
    if q[0] > 3.14159:
        q[0] = -3.14159 + q[0]
    elif q[0] <-3.14159:
        q[0] = -q[0]

    if q[1]-1.5708 > 2.5744:
        q[1] = -2.57445 + q[1]
    elif q[1]-1.5708 <-2.2689:
        q[1] = -q[1]

    if q[2] > 2.5307:
        q[2] = -2.5307 + q[2]
    elif q[2] < -2.5307:
        q[2] = -q[2]

    if q[3] > 4.7124:
        q[3] = -4.7124 + q[3]
    elif q[3] <-4.7124:
        q[3] = -q[3]

    if q[4] > 2.4435:
        q[4] = -2.4435 + q[4]
    elif q[4] <-2.0071:
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
    dh_param1 = np.array([0, 0.05, -pi/2]) # d a alpah
    dh_param2 = np.array([0, 0.425, 0])
    dh_param3 = np.array([0.05, 0, pi/2])
    dh_param4 = np.array([0.425, 0, -pi/2])
    dh_param5 = np.array([0, 0, pi/2])
    dh_param6 = np.array([0.1, 0, 0])
    dh_param7 = np.array([0.1027, 0.1911, 0])

    T12 = Homgm(dh_param1, q1, offset=0)
    T23 = Homgm(dh_param2, q2, offset=-pi/2)
    T34 = Homgm(dh_param3, q3, offset=pi/2)
    T45 = Homgm(dh_param4, q4, offset=0)
    T56 = Homgm(dh_param5, q5, offset=0)
    T67 = Homgm(dh_param6, q6, offset=0)
    T7E = Homgm(dh_param7, q7, offset=0)

    TF = T12@T23@T34@T45@T56@T67@T7E

    return TF


def jacobian_sym():
    '''
    Symbolic jacobian calculation
    '''
    q1, q2, q3, q4, q5, q6, q7= sym.symbols("q_1 q_2 q_3 q_4 q_5 q_6 q_7", real=True)  

    variables = [q1, q2, q3, q4, q5, q6, q7]
    var2 = q7
    dh_param1 = np.array([0, 0.05, -pi/2]) # d a alpah
    dh_param2 = np.array([0, 0.425, 0])
    dh_param3 = np.array([0.05, 0, pi/2])
    dh_param4 = np.array([0.425, 0, -pi/2])
    dh_param5 = np.array([0, 0, pi/2])
    dh_param6 = np.array([0.1, 0, 0])
    dh_param7 = np.array([0.1027, 0.1911, 0])

    T12 = Homgm_sym(dh_param1, q1, offset=0)
    T23 = Homgm_sym(dh_param2, q2, offset=-pi/2)
    T34 = Homgm_sym(dh_param3, q3, offset=pi/2)
    T45 = Homgm_sym(dh_param4, q4, offset=0)
    T56 = Homgm_sym(dh_param5, q5, offset=0)
    T67 = Homgm_sym(dh_param6, q6, offset=0)
    T7E = Homgm_sym(dh_param7, q7, offset=0)
    TF = T12@T23@T34@T45@T56@T67

    TF_Extend = T12@T23@T34@T45@T56@T67@T7E
    # # additional link

    R = TF_Extend[:3,:-1]
    jacobian = sym.Matrix([])

    T_d = sym.diff(TF_Extend, var2)
    T    = T_d[0:3, -1]
    R_d  = T_d[0:3, :-1]
    R_j  = R_d @ R.T 
    J_Extend = T.row_insert(3, sym.Matrix([R_j[2,1], R_j[0,2], R_j[1,0]])) # additional link

    for var in variables[:6]:
        T_d  = sym.diff(TF, var) 

        T    = T_d[0:3, -1]
        R_d  = T_d[0:3, :-1]
        R_j  = R_d @ R.T 

        J = T.row_insert(3, sym.Matrix([R_j[2,1], R_j[0,2], R_j[1,0]]))

        jacobian = jacobian.col_insert(len(jacobian), J)
    jacobian = jacobian.col_insert(len(jacobian), J_Extend) # additional link jacobian
    jacobian = sym.nsimplify(jacobian, tolerance=1e-5, rational=True) # remove near zero values

    return sym.lambdify([variables], jacobian, "numpy")

def jacobian(joint_params):
    variables = [*joint_params]
    return jacobian_sym_func(variables)

def plot_robot(q_parms):
    print(q_parms, q_parms.shape)
    print(np.rad2deg(q_parms))

    q1 = q_parms[0] 
    q2 = q_parms[1]
    q3 = q_parms[2]
    q4 = q_parms[3]
    q5 = q_parms[4]
    q6 = q_parms[5]
    q7 = 0

    dh_param1 = np.array([0, 0.05, -pi/2]) # d a alpah
    dh_param2 = np.array([0, 0.425, 0])
    dh_param3 = np.array([0.05, 0, pi/2])
    dh_param4 = np.array([0.425, 0, -pi/2])
    dh_param5 = np.array([0, 0, pi/2])
    dh_param6 = np.array([0.1, 0, 0])
    dh_param7 = np.array([0.1027, 0.1911, 0])

    T12 = Homgm(dh_param1, q1, offset=0)
    T23 = Homgm(dh_param2, q2, offset=-pi/2)
    T34 = Homgm(dh_param3, q3, offset=pi/2)
    T45 = Homgm(dh_param4, q4, offset=0)
    T56 = Homgm(dh_param5, q5, offset=0)
    T67 = Homgm(dh_param6, q6, offset=0)
    T7E = Homgm(dh_param7, q7, offset=0)

    TF = T12@T23@T34@T45@T56@T67@T7E

    T01 = np.eye(4)
    T02 = T01 @ T12
    T03 = T01 @ T12 @ T23
    T04 = T01 @ T12 @ T23 @ T34
    T05 = T01 @ T12 @ T23 @ T34 @ T45
    T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56
    T07 = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67
    T0E = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67 @ T7E
    p = T0E[:3,-1] # position 
    R = T0E[:3, :-1]
    rpy = euler_from_rotation(R)
    print("end pos : ", p[0], p[1], p[2], rpy)
    x_pos = [T01[0,-1], T02[0,-1], T03[0,-1], T04[0,-1], T05[0,-1], T06[0,-1], T07[0,-1], T0E[0,-1]]
    y_pos = [T01[1,-1], T02[1,-1], T03[1,-1], T04[1,-1], T05[1,-1], T06[1,-1], T07[1,-1], T0E[0,-1]]
    z_pos = [T01[2,-1], T02[2,-1], T03[2,-1], T04[2,-1], T05[2,-1], T06[2,-1], T07[2,-1], T0E[0,-1]]
 
    fig = go.Figure()
    fig.add_scatter3d(
        x=np.round(x_pos,2),
        y=np.round(y_pos,2),
        z=z_pos,
        line=dict( color='darkblue', width=15 ),
        hoverinfo="text",
        hovertext=[ f"joint {idx}: {q}" 
            for idx,q in 
              enumerate(np.round(np.rad2deg([ 0, q1, q2, q3, q4, q5, q6 ]),0)) ],
        marker=dict(
            size=10,
            color=[ np.linalg.norm([x,y,z]) for x,y,z in zip(x_pos, y_pos, z_pos) ],
            colorscale='Viridis',
        )
    )
    fig.layout=dict(
        width=1000,
        height=700,
        scene = dict( 
            camera=dict( eye={ 'x':-1.25, 'y':-1.25, 'z':2 } ),
            aspectratio={ 'x':1.25, 'y':1.25, 'z':1 },
            xaxis = dict( nticks=8, ),
            yaxis = dict( nticks=8 ),
            zaxis = dict( nticks=8 ),
            xaxis_title='Robot x-axis',
            yaxis_title='Robot y-axis',
            zaxis_title='Robot z-axis'),
        title=f"Robot in joint Configuration: {np.round(np.rad2deg(q_parms),0)} degrees",
        colorscale=dict(diverging="thermal")
    )
    pio.show(fig)

def simple_pseudo(q0, p_goal, time_step=1.2, max_iteration=500000, accuracy=0.001):

    goal_R = rotation_from_euler(p_goal[3:6])
    # Setting initial variables
    q_n0 = q0
    p = FK(q_n0)[:3,-1]
    R = FK(q_n0)[:3, :-1] # Rotation matrix
    p_goal[3] = goal_R[2][0]
    p_goal[4] = goal_R[2][1]
    p_goal[5] = goal_R[1][0] # p_goal[3,4,5] = R31, R32, R21
    p = np.array([ p[0], p[1], p[2], R[2][0], R[2][1], R[1][0]]) # shape miss match (6,1) # x,y,z R31, R32

    t_dot = p_goal - p
    δt = time_step
    i=0

    start_time = time.time()
    J_func    = dill.load(open(root+'\param_save\J_func_simp', "rb"))
    print("start runnign")
    q_dot = np.array([0, 0, 0, 0, 0, 0, 0])
    while True:
        # print(" ")
        # print(i, "/ 5000 ")
        # print(" ")
        if is_success(t_dot):
            print(f"Accuracy of {accuracy} reached")
            break
        q_n0 = Joint_limit_check(q_n0)
        q_n0 = q_n0 + (δt * q_dot)  
        q_n0[6] = 0
        p = FK(q_n0)[:3,-1]
        R = FK(q_n0)[:3,:-1] # Rotation matrix
        p = np.array([ p[0], p[1], p[2], R[2][0], R[2][1], R[1][0]]) # shape miss match (6,1) # x,y,z R31, R32

        T = find_T(R)
        invT = np.linalg.inv(T)
        J = J_func(q_n0)
        J_a = np.block([[np.eye(3), np.zeros((3,3))],[np.zeros((3, 3)), invT]]) @ J
        J_inv = np.linalg.pinv(J_a) 

        t_dot = p_goal - p
        q_dot = (J_inv @ t_dot)*0.1
        if is_success(t_dot):
            break
        i+=1
        if (i > max_iteration):
            print("No convergence")
            break

    end_time = time.time()
    print(f"Total time taken {np.round(end_time - start_time, 4)} seconds\n")

    return q_n0, p

def get_cnfs_null(method_fun, q0=np.deg2rad([0, 0, 0, 0, 0, 0, 0]), kwargs=dict()):
    # pos = np.array([0.674, 0.0431, -0.13358, 0.0028, 1.5708, 0])        #0.5sec
    # pos = np.array([0.674, 0.39453, -0.13358, 0.0028, 1.5708, 0])       #0.37sec
    # pos = np.array([0.674, -0.2752, 0.36959, 0.0028, 1.5708, 0])        #9.1sec
    # pos = np.array([0.67773, 0.41947, -0.1084, 0.0028, 1.5708, 0])      #0.37sec

    overallpos = posture_read()
    # print(overallpos[8])
    # q, p = method_fun(self.init_q, np.array(overallpos[8]), **kwargs)
    for i in range(len(overallpos)):
        q, p = method_fun(q0, np.array(overallpos[i]), **kwargs)
        print("pos", i, " ans : ", np.rad2deg(q))
        # print("arrived: ", p.T)
        # print(" ")

    end_time = time.time()

if __name__ == "__main__":
    # Length of Links in meters
    pi = np.pi
    pi_sym = sym.pi
    # q = np.deg2rad(np.array([-8.56, 40.89, 136.83, -8.56, -87.34, -0.11]))
    # plot_robot(q)
    # jacobian_sym_func = jacobian_sym()
    get_cnfs_null(method_fun=simple_pseudo, q0=np.deg2rad([10 ,10, 10, 10, 10, 10, 0]))