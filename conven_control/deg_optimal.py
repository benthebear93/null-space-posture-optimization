import os
import numpy as np
import pandas as pd
import sympy as sym

from gradient_save import Ktheta
sym.init_printing()

import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True, linewidth=200)

from spatialmath import *
import dill

def posture_read():

    df = pd.read_excel('random_position_test.xlsx', header=None, names=None, index_col=None)
    num_test = df.shape[0]

    print("number of test: ",  num_test)
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

    # def external_force(self): 
    #     '''
    #     External force 
    #     Output : Fx, Fy, Fz, Mx, My, Mz 
    #     shape  :(6x1)
    #     '''
    #     F = np.array([[3.0], [3.0], [-20.0], [0.0], [0.0], [0.0]])
    #     return F 

    def FK(self, joint_params):
        '''
        Joint variables consisting of 7 parameters
        ''' 

        joint_params = np.asarray(joint_params, dtype=float)
        q1, q2, q3, q4, q5, q6, q7 = joint_params

        q1 = deg2rad(q1)
        q2 = deg2rad(q2)
        q3 = deg2rad(q3)
        q4 = deg2rad(q4)
        q5 = deg2rad(q5)
        q6 = deg2rad(q6)
        q7 = deg2rad(q7)

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
        ''' 
        Joint limit check
        '''
        if q[0] > 180.0:
            q[0] = -180.0 + q[0]
        elif q[0] <-180.0:
            q[0] = -q[0]

        if q[1] > 147.5:
            q[1] = -147.5 + q[1]
        elif q[1] <-130.0:
            q[1] = -q[1]

        if q[2] > 145.0:
            q[2] = -145.0 + q[2]
        elif q[2] < -145.0:
            q[2] = -q[2]

        if q[3] > 270.0:
            q[3] = -270.0 + q[3]
        elif q[3] <-270.0:
            q[3] = -q[3]

        if q[4] > 140.0:
            q[4] = -140.0 + q[4]
        elif q[4] <-115.0:
            q[4] = -q[4]

        if q[5] > 270.0:
            q[5] = -270.0 + q[5]
        elif q[5] < -270.0:
            q[5] = -q[5]

        return q
    
    def is_success(self, error):
        
        accuracy = self.accuracy
        if abs(error[0]) < accuracy and abs(error[1]) < accuracy and abs(error[2]) < accuracy and abs(error[3]) < 0.01 and abs(error[4]) < 0.01: 
            return True

    def null_space_method(self,pos_num, q0, p_goal):
        '''
        Null space projection method
        '''
        max_iteration=500000
        Ktheta_inv = np.linalg.inv(self.Ktheta)

        goal_R = rotation_from_euler(p_goal[3:6])
        q_n0 = q0

        p = self.FK(q_n0)[:3,-1] # position 
        R = self.FK(q_n0)[:3, :-1] # Rotation matrix

        p_goal[3] = goal_R[2][0]
        p_goal[4] = goal_R[2][1]
        p_goal[5] = goal_R[1][0] # Goal position = [x, y, z R31, R32, R21]
        p = np.array([ [p[0]], [p[1]], [p[2]], [R[2][0]], [R[2][1]], [R[1][0]] ]) #shape (6,1) = [x, y, z R31, R32, R21]
        t_dot = p_goal[:5] - p[:5] # redundancy Rz remove (5,1)
        δt = self.time_step
        c = self.c

        i=0
        dx = []
        dy = []
        dz = []

        J_func    = dill.load(open(self.root+'\param_save\J_func_simp', "rb"))
        H_func    = dill.load(open(self.root+'\param_save\H_func_simp', "rb"))

        q_dot = np.array([0, 0, 0, 0, 0, 0, 0])
        while True:
            if self.is_success(t_dot):
                print("deviation : ", dxyz)
                print(f"Accuracy of {self.accuracy} reached")
                break
            q_n0 = self.Joint_limit_check(q_n0) 
            q_n0 = q_n0 + (δt * q_dot) 
            q_n0[6] = 0

            p = self.FK(q_n0)[:3,-1]
            R = self.FK(q_n0)[:3,:-1] # Rotation matrix
            p = np.array([ [p[0]], [p[1]], [p[2]], [R[2][0]], [R[2][1]], [R[1][0]] ]) #shape (5,1) = [x, y, z R31, R32, R21]
            
            T = find_T(R)
            invT = np.linalg.inv(T)
            # if i % 100 ==0:
            # print(pos_num, i, " q_n0: ", q_n0)
            # print(pos_num, i, " (δt * q_dot): ", (δt * q_dot))
            print(pos_num, i, " t_dot: ", t_dot.T)

            J = J_func(deg2rad(q_n0))
            J_a = np.block([[np.eye(3), np.zeros((3,3))],[np.zeros((3, 3)), invT]]) @ J
            J_na = J_a[:5]

            t_dot = p_goal[:5] - p[:5] # redundancy Rz remove (5,1)

            gH = H_func(deg2rad(q_n0))[0]  # gradient of cost function 
            gH = np.array([ [gH[0]],[gH[1]],[gH[2]],[gH[3]],[gH[4]],[gH[5]],[gH[6]] ])
            psd_J = J_na.T@ np.linalg.inv((J_na@J_na.T))# + c.T@np.eye(5) (7,5)
            K_  = 1.0
            Kn_ = 0.01
            print("psd_J @ t_dot\n", psd_J @ t_dot)
            print("(np.eye(7) - (psd_J @ J_na))@gH\n", (np.eye(7) - (psd_J @ J_na))@gH)
            # K_  = 2.5
            # Kn_ = 0.0002
            qdot = K_*psd_J @ t_dot -Kn_*(np.eye(7) - (psd_J @ J_na))@gH # 6x5 5x1    - (6x6-6x5 5x6) 7x1
            q_dot = np.array([qdot[0][0], qdot[1][0], qdot[2][0], qdot[3][0], qdot[4][0], qdot[5][0], 0]) # shape miss match (5,1) from (5,)
            J_temp =J_a[:,:6]@Ktheta_inv@J_a[:,:6].T
            dxyz = J_temp@self.F[:6] 

            dx.append(dxyz[0])
            dy.append(dxyz[1]) 
            dz.append(dxyz[2]) # for plot errors
            if self.is_success(t_dot):
                print("deviation : ", dxyz)
                break
            i+=1
            if (i > max_iteration):
                print("No convergence")
                break
        # print(f"to {np.round(p_goal,4)} :: time taken {np.round(end_time - start_time, 4)} seconds\n")
        plt.plot(dx, label="dx")
        plt.plot(dy, label="dy")
        plt.plot(dz, label="dz")
        plt.legend()
        plt.show()
        return q_n0, p, dxyz

    def get_cnfs_null(self, method_fun, kwargs=dict()):
        overallpos = posture_read()
        J1, J2, J3, J4, J5, J6, index, pos, dxyz = ([] for i in range(9))
        # q, p, d_xyz = method_fun(0, self.init_q, np.array(overallpos[0]), **kwargs)
        for i in range(len(overallpos)):
            q, p, d_xyz = method_fun(i, self.init_q, np.array(overallpos[i]), **kwargs)
            print("pos", i, " ans : ", np.rad2deg(q))
            index.append(i)
            J1.append(q[0])
            J2.append(q[1])
            J3.append(q[2])
            J4.append(q[3])
            J5.append(q[4])
            J6.append(q[5])
            pos.append(np.around(np.array(overallpos[i]), decimals=4))
            a = d_xyz[2]*d_xyz[2]+d_xyz[1]*d_xyz[1]+d_xyz[0]*d_xyz[0]
            dxyz.append(a)
        pos_record = pd.DataFrame({'J1':np.rad2deg(J1), 'J2':np.rad2deg(J2), 'J3':np.rad2deg(J3), 'J4':np.rad2deg(J4), 'J5':np.rad2deg(J5), 'J6':np.rad2deg(J6), 'pos':pos, 'deviation':dxyz}, index=index)
        pos_record.to_excel('optimized_result.xlsx', sheet_name='Sheet2', float_format="%.3f", header=True)

if __name__ == "__main__":
    # Length of Links in meters
    pi = np.pi
    pi_sym = sym.pi
    PosPlane = OptimalIK(0.1, 0.001, 50000)
    start = time.time()
    PosPlane.get_cnfs_null(method_fun=PosPlane.null_space_method)
    print("took : ", time.time() - start, " second")