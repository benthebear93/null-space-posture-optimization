import time
import numpy as np
import sympy as sym
sym.init_printing()
from math import atan2, sqrt
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from twist import *

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

Ktheta = np.diag(np.array([1.7, 5.9, 1.8, 0.29, 0.93, 0.49], dtype=np.float32)) # 3x3 
q_limit_high = np.array([3.14159, 2.5744, 2.5307, 4.7124, 2.4435, 4.7124], dtype=np.float32)
q_limit_low = np.array([-3.14159, -2.2689, -2.5307, -4.7124, -2.0071, -4.7124], dtype=np.float32)

def external_force(): 
    '''
    External force 
    Output : Fx, Fy 
    shape  :(2x1)
    '''

    F = np.array([[3.0], [3.0], [20.0], [0.0], [0.0], [0.0]])
    return F

def torque_variation(ext_force, joint_params):
    '''
    Torque variation due to the external force
    Input  : external force, joint angle
    Output : [dtorq1 dtorq2 dtorq3].T
    shape  : (3x1)
    '''

    J = jacobian(joint_params) # 2x3
    print("tq", type(J))
    dtorq = J.T@ext_force # 3x2 @ 2x1 = 3x1

def dcatersian(joint_params):
    ''' 
    Deviation on catersian space
    Input  : Joint anlge
    Output : deviation on x, y, z
    shape  :
    '''
    J = jacobian(joint_params) # 2x3
    # print("dc", type(J[0]))
    # print("J ", J[0])
    Ktheta_inv = np.linalg.inv(Ktheta) # 3x2
    print(J[0].shape, Ktheta_inv.shape)#, J.T.shape)
    J_temp = J[0]@Ktheta_inv@J[0].T #np.linalg.inv(JTinv@Ktheta@J.T)  # 2x3 @ 3x3 @ 3x2
    ext_force = external_force() # 2x2

    dxyz = J_temp@ext_force # 2x2 @ 2x2
    print("Deviation total : ", 0.5*(dxyz[0]**2+dxyz[1]**2))
    return dxyz

def euler_to_rotation(euler):
  R = Rz(euler[2]) @ Ry(euler[1]) @ Rx(euler[0])
  return R

def euler_angles(R, sndSol=True):
    '''
    Rotation to euler angle
    Input  : Rotation matrix
    Output : Rx, Ry, Rz
    '''

    rx = atan2(R[2,0], R[2,1])
    ry = atan2(sqrt(R[0,2]**2 + R[1,2]**2), R[2,2])
    rz = atan2(R[0,2], -R[1,2])

    return [rx, ry, rz]

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
    T67 = Homgm_sym(dh_param6, q6, offset=0)
    T7E = np.array([[1, 0, 0, 0.1911],
            [0, 1,  0, 0],
            [0, 0, 1, 0.1027],
            [0,0,0,1]])
    TF = T12@T23@T34@T45@T56@T67@T7E


    R = TF[:3,:-1]
    jacobian = sym.Matrix([])
    jacobian_null = sym.Matrix([])
    Hessian = sym.Matrix([])

    for var in variables:
        #print("calculating jacobian")
        T_d  = sym.diff(TF, var) 

        T    = T_d[0:3, -1] # translation?
        R_d  = T_d[0:3, :-1] #Rotation diff
        R_j  = R_d @ R.T  #Rotation jacobian

        J = T.row_insert(2, sym.Matrix([R_j[2,1], R_j[0,2], R_j[1,0]])) # [T_d; R_d] # jacobian calcuation for hessian
        J_null = T.row_insert(2, sym.Matrix([R_j[2,1], R_j[0,2]])) # null space control jacobian 
        jacobian = jacobian.col_insert(len(jacobian), J) # 6x1 translation + rotation diff 
        jacobian_null = jacobian_null.col_insert(len(jacobian), J_null) # 6x1 translation + rotation diff 

    F = external_force()
    Ktheta_inv = np.linalg.inv(Ktheta)
    target_jacobian = jacobian @ Ktheta_inv @ jacobian.T @ F # 2x3 @ 3x3 @ 3x2 @ 2x1 = 2x1
    H =0.5*(target_jacobian.row(0)@target_jacobian.row(0) + target_jacobian.row(1)@target_jacobian.row(1))

    for var in variables:
        #print("calculating hessian")
        T_d = sym.diff(H, var)
        Hessian = Hessian.col_insert(len(Hessian), T_d) # 3x1
    #print("H : ", Hessian)
    return sym.lambdify([variables], jacobian, "numpy"), sym.lambdify([variables], jacobian_null, "numpy"), sym.lambdify([variables], Hessian, "numpy") # Convert a SymPy expression into a function that allows for fast numeric evaluation.

def jacobian(joint_params):

    variables = [*joint_params]
    jacobian = jacobian_sym_func(variables)
    jacobian_null = jacobian_null_sym_func(variables)
    hessian = diff_jacobian_sym_func(variables)

    return jacobian, jacobian_null, hessian

# def diff_jacobian(joint_params):
#     variables = [*joint_params]

#     return diff_jacobian_sym_func(variables)

def null_space_method(q0, p_goal, time_step=0.1, max_iteration=5000, accuracy=0.01):

    #assert np.linalg.norm(p_goal[:2]) <= 0.85*np.sum([a1, a2, a3, a4]), "Robot Length constraint violated"
    '''
    p_goal : x, y, z, r, p, yaw
    nullspace contrl : x, y, z, r, p ( redundanct at yaw rotation )
    '''
    goal_R = euler_to_rotation(p_goal[3:6])

    q_n0 = q0
    p = FK(q_n0)[:3,-1] # position 
    R = FK(q_n0)[:3] # Rotation matrix

    p_goal[3] = goal_R[2][0]
    p_goal[4] = goal_R[2][1]
    p_goal[5] = goal_R[1][0]
    print("p_goal:", p_goal, "p_goal[3]: ", p_goal[3], "p_goal[4]: ", p_goal[4], "p_goal[5]", p_goal[5])

    p = np.array([ [p[0]], [p[1]], [p[2]], [R[2][0]], [R[2][1]]]) #, [R[1][0]]]) # shape miss match (6,1)

    t_dot = p_goal[:5] - p[:5] # redundancy remove (5,1)
    print("p_goal: ", p_goal, "p: ", p)
    print("t_dot:", t_dot, "type :", type(t_dot))

    q_n1 = q_n0
    δt = time_step

    i=0
    start_time = time.time()
    dx = []
    dy = []
    dz = []
    while True:
        # print(" ")
        # print(i, "/ 5000 ")
        # print(" ")
        if q_n0[0] > q_limit_high[0]:
            q_n0[0] = -q_limit_high[0] + q_n0[0]
        elif q_n0[0] <q_limit_low[0]:
            q_n0[0] = -q_n0[0]

        if q_n0[1] > q_limit_high[1]:
            q_n0[1] = -q_limit_high[1] + q_n0[1]
        elif q_n0[1] <q_limit_low[1]:
            q_n0[1] = -q_n0[1]

        if q_n0[2] > q_limit_high[2]:
            q_n0[2] = -q_limit_high[2] + q_n0[2]
        elif q_n0[2] <q_limit_low[2]:
            q_n0[2] = -q_n0[2]

        if q_n0[3] > q_limit_high[3]:
            q_n0[3] = -q_limit_high[3] + q_n0[3]
        elif q_n0[3] <q_limit_low[3]:
            q_n0[3] = -q_n0[3]

        if q_n0[4] > q_limit_high[4]:
            q_n0[4] = -q_limit_high[4] + q_n0[4]
        elif q_n0[4] <q_limit_low[4]:
            q_n0[4] = -q_n0[4]

        if q_n0[5] > q_limit_high[5]:
            q_n0[5] = -q_limit_high[5] + q_n0[5]
        elif q_n0[5] <q_limit_low[5]:
            q_n0[5] = -q_n0[5]

        # print(" ")
        # print("tdot : ", t_dot)
        if abs(t_dot[0][0]) < accuracy and abs(t_dot[1][0]) < accuracy and abs(t_dot[2][0]) < accuracy and abs(t_dot[3][0]) < accuracy and abs(t_dot[4][0]) < accuracy: 
            break
        start = time.time()
        p = FK(q_n0)[:3,-1] # position 
        R = FK(q_n0)[:3] # Rotation matrix
        end = time.time()
        print("#1 :", start - end)

        start = time.time()
        p = np.array([ [p[0]], [p[1]], [p[2]], [R[2][0]], [R[2][1]]]) #, [R[1][0]]])

        t_dot = p_goal[:5] - p[:5] # redundancy remove (5x1)
        # print(" ")
        # print("t dot", abs(t_dot[0][0]),abs(t_dot[1][0]),abs(t_dot[2][0]),abs(t_dot[3][0]),abs(t_dot[4][0]))

        J, J_n, H = jacobian(q_n0)
        end = time.time()
        start = time.time()
        print("#2 :", start - end)
        c = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001]) #, 0.001])
        #k = np.array([1.2, 1.2, 1.2, 1.2, 1.2, 1.2])

        psd_J = J_n.T@ np.linalg.inv((J_n@J_n.T + c.T@np.eye(5)))  
        qdot = psd_J @ t_dot - (np.eye(6) - (psd_J @ J_n))@H.T
        end = time.time()
        print("#3 :", start - end)
        # temp = psd_J @ t_dot
        # temp2 = (np.eye(3) - (psd_J @ J))
        # temp3 = H.T
        # temp4 = (np.eye(3) - (psd_J @ J))@H.T

        # print("psd_J                        :", psd_J.shape)
        # print("t_dot                        :", t_dot.shape)
        # print("qdot ", qdot)

        # print("psd_J @ t_dot                : ", temp.shape)
        # print("(np.eye(3) - (psd_J @ J))    : ", temp2.shape)
        # print("H.T                          : ", temp3.shape)
        # print("(np.eye(3) - (psd_J @ J))@H.T: ", temp4.shape)

        q_dot = np.array([qdot[0][0], qdot[1][0], qdot[2][0], qdot[3][0], qdot[4][0], qdot[5][0]]) # shape miss match (5,1) from (5,)
        q_n1 = q_n0 + (δt * q_dot)
        q_n0 = q_n1
        dxyz = dcatersian(q_n0)
        # JT_inv = np.linalg.inv(J.T)
        # print("JT_inv: ", JT_inv)
        # print("JT_inv@Ktheta@J.T: ", JT_inv@Ktheta@J.T)
        # J_temp = np.linalg.inv(JT_inv@Ktheta@J.T)
        # print("J_temp: ", J_temp)
        # F = external_force()
        # print("F: ", F)
        # dxyz = J_temp@F[:6] # 2x2 @ 2x2
        # print("dxyz: ", dxyz)
        # print("deviation x: ", dxyz[0])
        # print("deviation y: ", dxyz[1])
        # print("deviation z: ", dxyz[2])
        dx.append(dxyz[0])
        dy.append(dxyz[1]) 
        dz.append(dxyz[2]) # for plot errors
        # print("Deviation total : ", 0.5*(dxyz[0]**2+dxyz[1]**2))
        i+=1
        if (i > max_iteration):
            print("No convergence")
            break

    end_time = time.time()
    print(f"to {np.round(p_goal,2)} :: time taken {np.round(end_time - start_time, 4)} seconds\n")
      
    plt.plot(dx)
    plt.plot(dy)
    plt.plot(dz)
    plt.show()
    return q_n1

def plot_robot(q_parms):

    # q1 = q_parms[0][0] 
    # q2 = q_parms[0][1]
    # q3 = q_parms[0][2]
    # q4 = q_parms[0][3]
    # q5 = q_parms[0][4]
    # q6 = q_parms[0][5]

    q1 = q_parms[0][0] 
    q2 = q_parms[1][0]
    q3 = q_parms[2][0]
    q4 = q_parms[3][0]
    q5 = q_parms[4][0]
    q6 = q_parms[5][0]

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
    x_pos = [T01[0,-1], T02[0,-1], T03[0,-1], T04[0,-1], T05[0,-1], T06[0,-1], T0E[0,-1]]
    y_pos = [T01[1,-1], T02[1,-1], T03[1,-1], T04[1,-1], T05[1,-1], T06[1,-1], T0E[1,-1]]
    z_pos = [T01[2,-1], T02[2,-1], T03[2,-1], T04[2,-1], T05[2,-1], T06[2,-1], T0E[2,-1]]

    print(x_pos[0], y_pos[0], x_pos[1], y_pos[1], x_pos[2], y_pos[2])
    print("q1: ",np.rad2deg(q1), " q2: ", np.rad2deg(q2), " q3: ", np.rad2deg(q3), " q4: ", np.rad2deg(q4), " q5: ", np.rad2deg(q5), " q6: ", np.rad2deg(q6))
    
    print("x_pos: ", T0E[0,-1], " y_pos: ", T0E[1,-1], " z_pos: ", T0E[2,-1])
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
    pos = np.array([[0.674], [0.0431], [-0.1336], [-1.9], [1.5707], [-2.110]])
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
    # q = np.deg2rad([ [4.2563] ,[23.1508], [118.4877], [20.4948], [-53.4449], [-12.5557]])
    # plot_robot(q)
    jacobian_sym_func, jacobian_null_sym_func, diff_jacobian_sym_func = jacobian_sym()
    get_cnfs_null(method_fun=null_space_method, q0=np.deg2rad([5 ,5, 5, 5, 5, 5]))