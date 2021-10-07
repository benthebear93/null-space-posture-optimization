import time
import os
import numpy as np
import pandas as pd
import sympy as sym
sym.init_printing()

from math import atan2, sqrt
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True, linewidth=200)

from spatialmath import *
import pickle
import dill
def posture_read():
    # load_wb = load_workbook("C:/Users/UNIST/Desktop/stiffness_estimation/test_z.xlsx", data_only=True)
    df = pd.read_excel('random_position.xlsx', header=None, names=None, index_col=None)
    num_test = df.shape[0]

    print("number of test: ",  (num_test-1)/2)
    overall_posval =[]
    pos_val = []
    for i in range(0, num_test):
        for j in range(0, 6):
            a = [df.iloc[i][j]]
            pos_val.append(a)
        overall_posval.append(pos_val)
        pos_val = []
    print(overall_posval)
    return overall_posval


class OptimalIK:
    def __init__(self, time_step, accuracy, max_iteration):
        self.root = os.getcwd()
        self.Ktheta = np.diag(np.array([1.7, 5.9, 1.8, 0.29, 0.93 ,0.49]))
        self.time_step     = time_step
        self.accuracy      = accuracy
        self.max_iteration = max_iteration
        self.c         = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001]) #Tikhonov
        self.F         = np.array([[3.0], [3.0], [20.0], [0.0], [0.0], [0.0]])
        self.init_q    = np.array([0.1745, 0.1745, 0.1745, 0.1745, 0.1745, 0.1745, 0]) # avoid singurality
        # self.K         = np.array([[1.1], [1.1], [1.1], [0.1], [0.5], [1.1], [0]]) # avoid singurality

    def external_force(self): 
        '''
        External force 
        Output : Fx, Fy 
        shape  :(2x1)
        '''
        F = np.array([[3.0], [3.0], [20.0], [0.0], [0.0], [0.0]])
        return F 

    def FK(self, joint_params):
        """
        Joint variables consisting of 7 parameters
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
    
    def Joint_limit_check(self, q):
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
    
    def is_success(self, error):

        accuracy = self.accuracy
        if abs(error[0]) < accuracy and abs(error[1]) < accuracy and abs(error[2]) < accuracy and abs(error[3]) < 0.01 and abs(error[4]) < 0.01: 
            return True

    def null_space_method(self,pos_num, q0, p_goal):
        goal_R = rotation_from_euler(p_goal[3:6])
        q_n0 = q0

        p = self.FK(q_n0)[:3,-1] # position 
        R = self.FK(q_n0)[:3, :-1] # Rotation matrix

        p_goal[3] = goal_R[2][0]
        p_goal[4] = goal_R[2][1]
        p_goal[5] = goal_R[1][0] # Goal position = [x, y, z R31, R32, R21]
        p = np.array([ [p[0]], [p[1]], [p[2]], [R[2][0]], [R[2][1]], [R[1][0]] ]) #shape (5,1) = [x, y, z R31, R32, R21]

        #p = np.array([ p[0], p[1], p[2], R[2][0], R[2][1], R[1][0] ]) #shape (5,1) = [x, y, z R31, R32, R21]
        t_dot = p_goal[:5] - p[:5] # redundancy Rz remove (5,1)
        δt = self.time_step
        c = self.c
        i=0
        dx = []
        dy = []
        dz = []

        J_func    = dill.load(open(self.root+'\param_save\J_func_simp', "rb"))
        Jn_func   = dill.load(open(self.root+'\param_save\Jn_func_simp', "rb"))
        H_func    = dill.load(open(self.root+'\param_save\H_func_simp', "rb"))
        q_dot = np.array([0, 0, 0, 0, 0, 0, 0])
        while True:
            # print(" ")
            # print(i, "/ 5000 ")
            # print(" ")
            if self.is_success(t_dot):
                print(f"Accuracy of {self.accuracy} reached")
                break
            q_n0 = self.Joint_limit_check(q_n0)
            q_n0 = q_n0 + (δt * q_dot)
            q_n0[6] = 0

            p = self.FK(q_n0)[:3,-1]
            R = self.FK(q_n0)[:3,:-1] # Rotation matrix
            p = np.array([ [p[0]], [p[1]], [p[2]], [R[2][0]], [R[2][1]], [R[1][0]]]) #shape (5,1) = [x, y, z R31, R32, R21]
            
            T = find_T(R)
            invT = np.linalg.inv(T)
            # print(pos_num, i, " q_dot: ", q_dot)
            # print(pos_num, i, " t_dot: ", t_dot.T)
            J = J_func(q_n0)
            J_n  = Jn_func(q_n0)#[:,:6] # Jacobian null (6,5)
            J_a = np.block([[np.eye(3), np.zeros((3,3))],[np.zeros((3, 3)), invT]]) @ J
            J_na = J_a[:5]

            t_dot = p_goal[:5] - p[:5] # redundancy Rz remove (5,1)

            gH = H_func(q_n0)[0]  # gradient of cost function 
            psd_J = J_na.T@ np.linalg.inv((J_na@J_na.T + c.T@np.eye(5))) 
            qdot = psd_J @ t_dot - 0.001*(np.eye(7) - (psd_J @ J_na))@gH.T # 6x5 5x1    - (6x6-6x5 5x6) 7x1
            # print(i, "qdot:", qdot)
            q_dot = 2.5 * np.array([qdot[0][0], qdot[1][0], qdot[2][0], qdot[3][0], qdot[4][0], qdot[5][0], 0]) # shape miss match (5,1) from (5,)

            JT_inv = np.linalg.inv(J[:,:6].T)
            J_temp = np.linalg.inv(JT_inv@self.Ktheta@J[:,:6].T)
            dxyz = J_temp@self.F[:6] 

            dx.append(dxyz[0])
            dy.append(dxyz[1]) 
            dz.append(dxyz[2]) # for plot errors
            i+=1
        # print(f"to {np.round(p_goal,4)} :: time taken {np.round(end_time - start_time, 4)} seconds\n")
        # plt.plot(dx, label="dx")
        # plt.plot(dy, label="dy")
        # plt.plot(dz, label="dz")
        # plt.legend()
        # plt.show()
        return q_n0, p

    def get_cnfs_null(self, method_fun, kwargs=dict()):
        pos1 = np.array([[0.674], [0.0431], [-0.13358], [0.0028], [1.5708], [0]]) 
        pos2 = np.array([[0.674], [0.39453], [-0.13358], [0.0028], [1.5708], [0]])
        pos3 = np.array([[0.674], [-0.2752], [ 0.36959], [0.0028], [1.5708], [0]]) 
        pos4 = np.array([[0.67773], [0.41947], [ -0.1084], [0.0028], [1.5708], [0]]) 

        overallpos = posture_read()
        # print(overallpos[8])
        # q, p = method_fun(self.init_q, np.array(overallpos[8]), **kwargs)
        for i in range(len(overallpos)):
            q, p = method_fun(i, self.init_q, np.array(overallpos[i]), **kwargs)
            print("pos", i, " ans : ", np.rad2deg(q))
            # print("arrived: ", p.T)
            # print(" ")
        self.plot_robot(q)

    def plot_robot(self, q_parms):
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
if __name__ == "__main__":
    # Length of Links in meters
    pi = np.pi
    pi_sym = sym.pi
    PosPlane = OptimalIK(0.1, 0.001, 50000)
    PosPlane.get_cnfs_null(method_fun=PosPlane.null_space_method)