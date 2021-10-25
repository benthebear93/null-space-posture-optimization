import os
import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import sympy as sym

from gradient_save import Ktheta
sym.init_printing()

import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True, linewidth=200)

from spatialmath import *
import dill

def fk(joint_params):
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
    T7E = Homgm(dh_param7, 0, offset=0)

    TF = T12@T23@T34@T45@T56@T67@T7E

    return TF

def deviation(q):

    q = np.append(q, 0)
    Ktheta = np.diag(np.array([1.7, 5.9, 1.8, 0.29, 0.93 ,0.49]))
    Ktheta_inv = np.linalg.inv(Ktheta)
    F      = np.array([-6.0, -6.0, 40.0, 0.0, 0.0, 0.0])
    R = fk(q)[:3, :-1] # Rotation matrix

    T = find_T(R)
    invT = np.linalg.inv(T)

    root = '/home/benlee/Desktop/git/null-space-posture-optimization/conven_control/' #os.getcwd()
    # root = os.getcwd()
    J_func    = dill.load(open(root+'/param_save/J_func_simp', "rb"))
    J = J_func(q)
    J_a = np.block([[np.eye(3), np.zeros((3,3))],[np.zeros((3, 3)), invT]]) @ J
    J_inv = np.linalg.pinv(J_a) 
    J_temp =J[:,:6]@Ktheta_inv@J[:,:6].T
    dxyz = J_temp@F[:6]
    return dxyz

def cal_dev(filename):
    dx,dy,dz = ([] for i in range(3))
    df = pd.read_excel(filename, header=None, names=None, index_col=None)
    #df.columns = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'POS', 'dx','dy', 'dz']
    number_pose = len(df)
    for i in range(1, number_pose):
        q = np.array(df.iloc[i][1:7])
        q = deg2rad(q)
        dxyz = deviation(q)
        dx.append(dxyz[0])
        dy.append(dxyz[1])
        dz.append(0.5*(dxyz[0]*dxyz[0]+dxyz[1]*dxyz[1]))
    print(dx, len(dx))
    print(df.index)
    dx.append(0)
    dy.append(0)
    dz.append(0)
    df.insert(8,'dx', dx)
    df.insert(9,'dy', dy)
    df.insert(10,'dz', dz)
    #df.columns = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'POS', 'dx']
    df.to_excel(filename, index=False, float_format="%.3f", header=False)

if __name__=="__main__":
    # q = np.deg2rad(np.array([-39.076,	45.97,	107.678	,-28.401,	-65.783,	12.138]))
    # dxyz = deviation(q)
    # print("opt :", dxyz)

    # q = np.deg2rad(np.array([-39.076,	45.97,	107.678	,-28.401,	-65.783+90,	12.138-90]))
    # dxyz = deviation(q)
    # print("non opt:", dxyz)

    # # q = np.deg2rad(np.array([-24.123,	42.317,	118.672,	-9.676,	-71.182,	3.023]))
    # # dxyz = deviation(q)
    # # print("opt :", dxyz)

    # # q = np.deg2rad(np.array([ -30.275,	45.5,	115.93,	148.417,	74.009,	189.556]))
    # # dxyz = deviation(q)
    # # print("non opt:", dxyz)
    
    # # q = np.deg2rad(np.array([-4.994,	41.283,	121.859,	13.089,	-73.441,	-3.882]))
    # # dxyz = deviation(q)
    # # print("opt :", dxyz)

    # # q = np.deg2rad(np.array([-13.184,	43.719,	122.137,	166.367,	76.201,	183.25]))
    # # dxyz = deviation(q)
    # # print("non opt:", dxyz)
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    cal_dev(root+'/data/ros_non_optimized_curved.xlsx')
    # cal_dev(root+'/data/optimized_result_fast_test.xlsx')