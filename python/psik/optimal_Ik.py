import time
import numpy as np
import sympy as sym
sym.init_printing()
from math import atan2, sqrt
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from twist import *
from plot_robot import *
import pickle
import dill

class OptimalIK:
    def __init__(self, time_step, accuracy, max_iteration):
        self.Ktheta = np.diag(np.array([1.7, 5.9, 1.8, 0.29, 0.93 ,0.49]))
        self.time_step     = time_step
        self.accuracy      = accuracy
        self.max_iteration = max_iteration
        self.c         = np.array([0.001, 0.001, 0.001, 0.001, 0.001]) #Tikhonov
        self.F         = np.array([[3.0], [3.0], [20.0], [0.0], [0.0], [0.0]])
        self.init_q    = np.array([0.1745, 0.1745, 0.1745, 0.1745, 0.1745, 0.1745]) # avoid singurality
        self.K         = np.array([[0.5], [0.5], [0.5], [0.5], [0.5], [0.5]]) # avoid singurality

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
    
    def Joint_limit_check(self, q):
        if q[0] > 3.14159:
            q[0] = -3.14159 + q[0]
        elif q[0] <-3.14159:
            q[0] = -q[0]

        if q[1] > 2.5744:
            q[1] = -2.57445 + q[1]
        elif q[1] <-2.2689:
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
        if abs(error[0]) < accuracy and abs(error[1]) < accuracy and abs(error[2]) < accuracy and abs(error[3]) < accuracy and abs(error[4]) < accuracy: 
            return True

    def null_space_method(self, q0, p_goal):

        goal_R = rotation_from_euler(p_goal[3:6])

        p_goal[3] = goal_R[2][0]
        p_goal[4] = goal_R[2][1]
        p_goal[5] = goal_R[1][0] # Goal position = [x, y, z R31, R32, R21]

        print("p_goal: ", p_goal, p_goal.shape)

        q_n0 = q0

        p = self.FK(q_n0)[:3,-1] # position 
        R = self.FK(q_n0)[:3] # Rotation matrix

        p = np.array([ [p[0]], [p[1]], [p[2]], [R[2][0]], [R[2][1]], [R[1][0]] ]) #shape (5,1) = [x, y, z R31, R32, R21]
        #p = np.array([ p[0], p[1], p[2], R[2][0], R[2][1], R[1][0] ]) #shape (5,1) = [x, y, z R31, R32, R21]
        print("p: ", p, p.shape)
        t_dot = p_goal[:5] - p[:5] # redundancy Rz remove (5,1)
        print("t_dot: ", t_dot, t_dot.shape)
        q_n1 = q_n0
        δt = self.time_step
        c = self.c
        i=0
        dx = []
        dy = []
        dz = []
        start_time = time.time()
        J_func    = dill.load(open("J_func_simp", "rb"))
        Jn_func   = dill.load(open("Jn_func_simp", "rb"))
        H_func    = dill.load(open("H_func_simp", "rb"))
        while True:
            # print(" ")
            # print(i, "/ 5000 ")
            # print(" ")
            if self.is_success(t_dot):
                break
            q_n0 = self.Joint_limit_check(q_n0)

            p = self.FK(q_n0)[:3,-1]
            R = self.FK(q_n0)[:3] # Rotation matrix
            p = np.array([ [p[0]], [p[1]], [p[2]], [R[2][0]], [R[2][1]], [R[1][0]] ]) #shape (5,1) = [x, y, z R31, R32, R21]
            t_dot = p_goal[:5] - p[:5] # redundancy remove (5,1)
            #print("error: ", t_dot)
            J_n  = Jn_func(q_n0) # Jacobian null (6,5)
            J = J_func(q_n0)
            if i ==0:
                print("q ", q_n0)
                print("p: ", p)
                print("t_dot: ", t_dot)
                print("Jacovian ", J)
            gH = H_func(q_n0)  # gradient of cost function 
            psd_J = J_n.T@ np.linalg.inv((J_n@J_n.T + c.T@np.eye(5))) 
            qdot = psd_J @ t_dot @self.K.T - 1.5*(np.eye(6) - (psd_J @ J_n))@gH.T
            q_dot = np.array([qdot[0][0], qdot[1][0], qdot[2][0], qdot[3][0], qdot[4][0], qdot[5][0]]) # shape miss match (5,1) from (5,)
            
            q_n1 = q_n0 + (δt * q_dot)
            q_n0 = q_n1

            JT_inv = np.linalg.inv(J.T)
            J_temp = np.linalg.inv(JT_inv@self.Ktheta@J.T)
            dxyz = J_temp@self.F[:6] 

            dx.append(dxyz[0])
            dy.append(dxyz[1]) 
            dz.append(dxyz[2]) # for plot errors
            i+=1
            if (i > self.max_iteration):
                print("No convergence")
                break
        end_time = time.time()
        print(f"to {np.round(p_goal,2)} :: time taken {np.round(end_time - start_time, 4)} seconds\n")
        plt.plot(dx, label="dx")
        plt.plot(dy, label="dy")
        plt.plot(dz, label="dz")
        plt.legend()
        plt.show()
        return q_n0

    def get_cnfs_null(self, method_fun, kwargs=dict()):
        pos = np.array([[0.523], [-0.244], [0.425], [-1.5708], [-1.0472], [1.5708]])
        start_time = time.time()

        q = method_fun(self.init_q, pos, **kwargs)

        end_time = time.time()
        print(f"\n{np.round(end_time-start_time, 1)} seconds : Total time using {method_fun.__name__} \n")
        if kwargs: print(f"\nParameters used: {kwargs}")

        plot_robot(q)

if __name__ == "__main__":
    # Length of Links in meters
    pi = np.pi
    pi_sym = sym.pi
    PosPlane = OptimalIK(0.1, 0.01, 50000)
    PosPlane.get_cnfs_null(method_fun=PosPlane.null_space_method)