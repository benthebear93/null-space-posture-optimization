import time
import numpy as np
import sympy as sym
sym.init_printing()
from math import atan2, sqrt
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from twist import *
import pickle

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import dill

Ktheta = np.diag(np.array([1.7, 5.9, 1.8, 0.29, 0.93 ,0.49], dtype=np.float64)) # 3x3 

def external_force(): 
    '''
    External force 
    Output : Fx, Fy 
    shape  :(2x1)
    '''
    F = np.array([[3.0], [3.0], [20.0], [0.0], [0.0], [0.0]])
    return F

def FK(joint_params):
    '''
    Joint variables consisting of 7 parameters
    '''
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
    
def jacobian(joint_params):

    variables = [*joint_params]
    jacobian = jacobian_sym_func(variables)
    hessian = diff_jacobian_sym_func(variables)

    return jacobian, hessian

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
        print("calculating jacobian")
        T_d  = sym.diff(TF, var) 

        T    = T_d[0:3, -1]  # translation?  
        R_d  = T_d[0:3, :-1] # Rotation diff 
        R_j  = R_d @ R.T     # Rotation jacobian

        J = T.row_insert(3, sym.Matrix([R_j[2,1], R_j[0,2], R_j[1,0]])) # [T_d; R_d] # jacobian calcuation for hessian
        J_null = T.row_insert(2, sym.Matrix([R_j[2,1], R_j[0,2]])) # null space control jacobian 
        jacobian = jacobian.col_insert(len(jacobian), J) # 6x1 translation + rotation diff 
        jacobian_null = jacobian_null.col_insert(len(jacobian_null), J_null) # 6x1 translation + rotation diff 
    
    jacobian = sym.nsimplify(jacobian,tolerance=1e-5,rational=True)
    jacobian_null = sym.nsimplify(jacobian_null,tolerance=1e-5,rational=True)

    print(jacobian_null[0])
    F = external_force()
    Ktheta_inv = np.linalg.inv(Ktheta)
    target_jacobian = jacobian @ Ktheta_inv @ jacobian.T @ F # 2x3 @ 3x3 @ 3x2 @ 2x1 = 2x1

    H =0.5*(target_jacobian.row(2)@target_jacobian.row(2))# + target_jacobian.row(1)@target_jacobian.row(1))

    for var in variables:
        print("calculating hessian")
        T_d = sym.diff(H, var)
        Hessian = Hessian.col_insert(len(Hessian), T_d) # 3x1

    Hessian = sym.nsimplify(Hessian,tolerance=1e-5,rational=True)
    print(jacobian.shape, jacobian_null.shape, Hessian.shape)
    with open('Jacobian.txt','wb') as f:
        pickle.dump(jacobian,f)
    with open('Jacobian_null.txt','wb') as f:
        pickle.dump(jacobian_null,f)
    with open('hessian.txt','wb') as f:
        pickle.dump(Hessian,f)
    return sym.lambdify([variables], jacobian_null, "numpy"), sym.lambdify([variables], Hessian, "numpy") # Convert a SymPy expression into a function that allows for fast numeric evaluation.

def get_cnfs_null(method_fun, q0=np.deg2rad([0, 0, 0, 0, 0, 0]), kwargs=dict()):
    # x = np.array([1.1464])
    # y = np.array([0.1999999])
    # #z = np.array([0])
    rob_cnfs = []
    pos = np.array([[0.523], [-0.244], [0.425], [-1.5708], [-1.0472], [1.5708]])
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
    jacobian_sym_func = jacobian_sym()
    get_cnfs_null(method_fun=null_space_method, q0=np.deg2rad([5 ,5, 5, 5, 5, 5]))