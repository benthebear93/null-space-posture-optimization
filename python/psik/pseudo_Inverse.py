import time
import numpy as np
import sympy as sym
sym.init_printing()
from math import atan2, sqrt
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from twist import *
import dill

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

def FK(joint_params):
    """
    Joint variables consisting of 7 parameters
    """
    joint_params = np.asarray(joint_params, dtype=float)
    q1, q2, q3, q4, q5, q6 = joint_params
    dh_param1 = np.array([0, 0.05, -pi/2]) # d a alpah
    dh_param2 = np.array([0, 0.425, 0])
    dh_param3 = np.array([0.05, 0, pi/2])
    dh_param4 = np.array([0.425, 0, -pi/2])
    dh_param5 = np.array([0, 0, pi/2])
    dh_param6 = np.array([0.1, 0, 0])

    T12 = Homgm(dh_param1, q1, offset=0)
    T23 = Homgm(dh_param2, q2, offset=-pi/2)
    T34 = Homgm(dh_param3, q3, offset=pi/2)
    T45 = Homgm(dh_param4, q4, offset=0)
    T56 = Homgm(dh_param5, q5, offset=0)
    T67 = Homgm(dh_param6, q6, offset=0)
    T7E = np.array([[1, 0, 0, 0.1911],
                [0, 1,  0, 0],
                [0, 0, 1, 0.1027],
                [0,0,0,1]])
    TF = T12@T23@T34@T45@T56@T67@T7E

    return TF

def jacobian_sym():
    '''
    Symbolic jacobian calculation
    '''
    q1, q2, q3, q4, q5, q6 = sym.symbols("q_1 q_2 q_3 q_4 q_5 q_6", real=True)  

    variables = [q1, q2, q3, q4, q5, q6]

    dh_param1 = np.array([0, 0.05, -pi/2]) # d a alpah
    dh_param2 = np.array([0, 0.425, 0])
    dh_param3 = np.array([0.05, 0, pi/2])
    dh_param4 = np.array([0.425, 0, -pi/2])
    dh_param5 = np.array([0, 0, pi/2])
    dh_param6 = np.array([0.1, 0, 0])

    T12 = Homgm_sym(dh_param1, q1, offset=0)
    T23 = Homgm_sym(dh_param2, q2, offset=-pi/2)
    T34 = Homgm_sym(dh_param3, q3, offset=pi/2)
    T45 = Homgm_sym(dh_param4, q4, offset=0)
    T56 = Homgm_sym(dh_param5, q5, offset=0)
    T67 = Homgm(dh_param6, q6, offset=0)
    T7E = np.array([[1, 0, 0, 0.1911],
                [0, 1,  0, 0],
                [0, 0, 1, 0.1027],
                [0,0,0,1]])
    TF = T12@T23@T34@T45@T56@T67@T7E



    R = TF[:3,:-1]
    jacobian = sym.Matrix([])

    for var in variables:
        T_d  = sym.diff(TF, var) 

        T    = T_d[0:3, -1] # translation?
        R_d  = T_d[0:3, :-1] #Rotation diff
        R_j  = R_d @ R.T  #Rotation jacobian
        # print("var : ", var)
        # print("R_j", R_j)

        J = T.row_insert(3, sym.Matrix([R_j[2,1], R_j[0,2], R_j[1,0]])) # [T_d; R_d]
        jacobian = jacobian.col_insert(len(jacobian), J) # 6x1 translation + rotation diff 
    jacobian = sym.nsimplify(jacobian,tolerance=1e-5,rational=True)
  
    return sym.lambdify([variables], jacobian, "numpy") # Convert a SymPy expression into a function that allows for fast numeric evaluation.

def jacobian(joint_params):
    variables = [*joint_params]
    return jacobian_sym_func(variables)

def simple_pseudo(q0, p_goal, time_step=0.01, max_iteration=100000, accuracy=0.01):

  print(p_goal[3:6])
  goal_R = rotation_from_euler(p_goal[3:6])

  q_n0 = q0
  print("q0: ", q_n0)
  temp = FK(q_n0)
  p = FK(q_n0)[:3,-1] # position 
  R = FK(q_n0)[:3, :-1] # Rotation matrix
  print("HM", temp)
  print("Ro: ", R)

  p_goal[3] = goal_R[2][0]
  p_goal[4] = goal_R[2][1]
  p_goal[5] = goal_R[1][0] # p_goal[3,4,5] = R31, R32, R21
  p = np.array([ p[0], p[1], p[2], R[2][0], R[2][1], R[1][0]]) # shape miss match (6,1) # x,y,z R31, R32

  t_dot = p_goal - p
  print("p_goal: ", p_goal, " p: ", p)
  print("t_dot:", t_dot, "type :", type(t_dot))

  q_n1 = q_n0
  δt = time_step

  i=0
  i = 0
  Tt = np.block([np.eye(6)])
  start_time = time.time()
  J_func=dill.load(open("J_func_simp", "rb"))
  while True:
    print(" ")
    print(i, "/ 5000 ")
    print(" ")
    if q_n0[0] > 3.14159:
        q_n0[0] = -3.14159 + q_n0[0]
    elif q_n0[0] <-3.14159:
        q_n0[0] = -q_n0[0]

    if q_n0[1] > 2.5744:
        q_n0[1] = -2.57445 + q_n0[1]
    elif q_n0[1] <-2.2689:
        q_n0[1] = -q_n0[1]

    if q_n0[2] > 2.5307:
        q_n0[2] = -2.5307 + q_n0[2]
    elif q_n0[2] < -2.5307:
        q_n0[2] = -q_n0[2]

    if q_n0[3] > 4.7124:
        q_n0[3] = -4.7124 + q_n0[3]
    elif q_n0[3] <-4.7124:
        q_n0[3] = -q_n0[3]

    if q_n0[4] > 2.4435:
        q_n0[4] = -2.4435 + q_n0[4]
    elif q_n0[4] <-2.0071:
        q_n0[4] = -q_n0[4]

    if q_n0[5] > 4.7124:
        q_n0[5] = -4.7124 + q_n0[5]
    elif q_n0[5] < -4.7124:
        q_n0[5] = -q_n0[5]

    if abs(t_dot[0]) < accuracy and abs(t_dot[1]) < accuracy and abs(t_dot[2]) < accuracy and abs(t_dot[3]) < accuracy and abs(t_dot[4]) < accuracy and abs(t_dot[5]) < accuracy: 
      print(f"Accuracy of {accuracy} reached")
      break
    p = FK(q_n0)[:3,-1]
    R = FK(q_n0)[:3,:-1] # Rotation matrix
    p = np.array([ p[0], p[1], p[2], R[2][0], R[2][1], R[1][0]]) # shape miss match (6,1) # x,y,z R31, R32

    t_dot = p_goal - p # redundancy remove (5,1) 
    #print(i ,"t_dot ", t_dot)
    # print("p_goal: ", p_goal, " p: ", p)
    # print("tdot", t_dot)
    J = J_func(q_n0)
    J_inv = np.linalg.pinv( (Tt @ J) )  # inv jacobian
    # print("tdot : ", t_dot)
    q_dot = J_inv @ t_dot
    q_n1 = q_n0 + (δt * q_dot)  
    q_n0 = q_n1
    i+=1
    if (i > max_iteration):
      print("No convergence")
      break

  end_time = time.time()
  print(f"Total time taken {np.round(end_time - start_time, 4)} seconds\n")

  return q_n0

def plot_robot(q_parms):

    # q1 = q_parms[0] 
    # q2 = q_parms[1]
    # q3 = q_parms[2]
    # q4 = q_parms[3]
    # q5 = q_parms[4]
    # q6 = q_parms[5]


    q1 = q_parms[0][0] 
    q2 = q_parms[0][1]
    q3 = q_parms[0][2]
    q4 = q_parms[0][3]
    q5 = q_parms[0][4]
    q6 = q_parms[0][5]

    # DH parameter [ d a alpah ]
    dh_param1 = np.array([0, 0.05, -pi/2]) 
    dh_param2 = np.array([0, 0.425, 0])
    dh_param3 = np.array([0.05, 0, pi/2])
    dh_param4 = np.array([0.425, 0, -pi/2])
    dh_param5 = np.array([0, 0, pi/2])
    dh_param6 = np.array([0.1, 0, 0])

    T12 = Homgm(dh_param1, q1, offset=0)
    T23 = Homgm(dh_param2, q2, offset=-pi/2)
    T34 = Homgm(dh_param3, q3, offset=pi/2)
    T45 = Homgm(dh_param4, q4, offset=0)
    T56 = Homgm(dh_param5, q5, offset=0)
    T67 = Homgm(dh_param6, q6, offset=0)
    T7E = np.array([[1, 0, 0, 0.1911],
                [0, 1,  0, 0],
                [0, 0, 1, 0.1027],
                [0,0,0,1]])
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

    print("end pos : ", p[0], p[1], p[2])
    x_pos = [T01[0,-1], T02[0,-1], T03[0,-1], T04[0,-1], T05[0,-1], T06[0,-1], T07[0,-1], T0E[0,-1]]
    y_pos = [T01[1,-1], T02[1,-1], T03[1,-1], T04[1,-1], T05[1,-1], T06[1,-1], T07[1,-1], T0E[0,-1]]
    z_pos = [T01[2,-1], T02[2,-1], T03[2,-1], T04[2,-1], T05[2,-1], T06[2,-1], T07[2,-1], T0E[0,-1]]
    print(" ")
    print("q1: ",np.rad2deg(q1), " q2: ", np.rad2deg(q2), " q3: ", np.rad2deg(q3), " q4: ", np.rad2deg(q4), " q5: ", np.rad2deg(q5), " q6: ", np.rad2deg(q6))
    print(" ")
    rpy = euler_from_rotation(T0E[:3])
    print("x_pos: ", T0E[0,-1], " y_pos: ", T0E[1,-1], " z_pos: ", T0E[2,-1], "roll: ", rpy[0], "pitch: ", rpy[1], "yaw: ", rpy[2])
    fig = go.Figure()
    fig.add_scatter3d(
        x=np.round(x_pos,2),
        y=np.round(y_pos,2),
        z=z_pos,
        line=dict( color='darkblue', width=15 ),
        hoverinfo="text",
        hovertext=[ f"joint {idx}: {q}" 
            for idx,q in 
              enumerate(np.round(np.rad2deg([ 0, q1, q2, q3, q4, q5, q6]),0)) ],
        marker=dict(
            size=10,
            color=[ np.linalg.norm([x,y,z]) for x,y,z in zip(x_pos, y_pos, z_pos) ],
            colorscale='Viridis',
        )
    )
    fig.layout=dict(
        width=1000,
        height=1000,
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

def get_cnfs_null(method_fun, q0=np.deg2rad([0, 0, 0, 0, 0, 0]), kwargs=dict()):
    # x = np.array([1.1464])
    # y = np.array([0.1999999])
    # #z = np.array([0])
    rob_cnfs = []
    pos = np.array([0.550, -0.0035, 0.2959, -2.3049, 1.5602, -2.3050])
    start_time = time.time()
    # for (i, j) in zip (x, y):# k, z
    #   pos = [i, j] # k

    q = method_fun(q0, pos, **kwargs)
    rob_cnfs.append(q)

    end_time = time.time()
    print(f"\n{np.round(end_time-start_time, 1)} seconds : Total time using {method_fun.__name__} \n")
    if kwargs: print(f"\nParameters used: {kwargs}")

    plot_robot(rob_cnfs)

if __name__ == "__main__":
    # Length of Links in meters
    pi = np.pi
    pi_sym = sym.pi
    # q = np.deg2rad(np.array([-8.56, 40.89, 136.83, -8.56, -87.34, -0.11]))
    # plot_robot(q)
    get_cnfs_null(method_fun=simple_pseudo, q0=np.deg2rad([10 ,10, 10, 10, 10, 10]))