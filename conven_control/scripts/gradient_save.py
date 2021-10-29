import os
import time
import numpy as np
import sympy as sym
sym.init_printing()

from spatialmath import *
import pickle
import dill

Ktheta = np.diag(np.array([1.7, 5.9, 1.8, 0.29, 0.93 ,0.49])) # 3x3 
root = os.getcwd()

def external_force(): 
    F = np.array([[-6.0], [-6.0], [40.0], [0.0], [0.0], [0.0]])
    return F

def FK(joint_params):
    '''
    6 dof Forward kinematic with additional last link(spindle)
    '''
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
    # T7E = np.array([[1, 0, 0, 0.1911],
    #         [0, 1,  0, 0],
    #         [0, 0, 1, 0.1027],
    #         [0,0,0,1]])
    TF = T12@T23@T34@T45@T56@T67#@T7E
    print(TF)
    TF_Extend = T12@T23@T34@T45@T56@T67@T7E

    
    R = TF[:3,:-1]
    jacobian      = sym.Matrix([])
    jacobian_ori  = sym.Matrix([])
    jacobian_null = sym.Matrix([])
    target_f      = sym.Matrix([])

    T_d  = sym.diff(TF_Extend, var2) # additional link 
    T    = T_d[0:3, -1]
    R_d  = T_d[0:3, :-1]
    R_j  = R_d @ R.T 
    J_Extend   = T.row_insert(3, sym.Matrix([ R_j[2,1], R_j[2,0], R_j[1,0] ]))
    J_n_Extend = T.row_insert(2, sym.Matrix([ R_j[2,1], R_j[2,0] ]))

    for var in variables[:6]:
        print("calculating jacobian")
        T_d  = sym.diff(TF, var) 

        T    = T_d[0:3, -1]  # translation?  
        R_d  = T_d[0:3, :-1] # Rotation diff 
        R_j  = R_d @ R.T     # Rotation jacobian

        J = T.row_insert(3, sym.Matrix([R_j[2,1], R_j[2,0], R_j[1,0]])) # [T_d; R_d] # jacobian calcuation for target_f
        J_null = T.row_insert(2, sym.Matrix([R_j[2,1], R_j[2,0]])) # null space control jacobian 
        jacobian = jacobian.col_insert(len(jacobian), J) # 6x1 translation + rotation diff 
        jacobian_null = jacobian_null.col_insert(len(jacobian_null), J_null) # 6x1 translation + rotation diff 

    jacobian_ori  = jacobian #for target function
    jacobian      = jacobian.col_insert(len(jacobian), J_Extend) # additional link jacobian
    jacobian_null = jacobian_null.col_insert(len(jacobian_null), J_n_Extend) # additional link jacobian

    jacobian      = sym.nsimplify(jacobian,tolerance=1e-5,rational=True)
    jacobian_null = sym.nsimplify(jacobian_null,tolerance=1e-5,rational=True)

    F = external_force()
    Ktheta_inv = np.linalg.inv(Ktheta)
    target_jacobian = (jacobian_ori @ Ktheta_inv @ jacobian_ori.T) @ F # 2x3 @ 3x3 @ 3x2 @ 2x1 = 2x1
    print("  ")
    print("shape :", target_jacobian.shape, type(target_jacobian))
    print("  ")

    norm = 0.5*(target_jacobian.row(0)*target_jacobian.row(0)+target_jacobian.row(1)*target_jacobian.row(1))
    print("calculationg norm")
    H =sym.Matrix([norm])#+target_jacobian.row(1)@target_jacobian.row(1)+target_jacobian.row(0)@target_jacobian.row(0))# target function for optimization 
    print("calculationg norm")
    print(type(H), H.shape)
    for var in variables:
        print("calculating gradient")
        T_d = sym.diff(H, var)
        target_f = target_f.col_insert(len(target_f), T_d) # 3x1

    target_f = sym.nsimplify(target_f,tolerance=1e-5,rational=True)
    print("  ")
    print("shape :", jacobian.shape, jacobian_null.shape, target_f.shape)
    print("  ")
    with open(root+'/Jacobian.txt','wb') as f:
        pickle.dump(jacobian,f)
    with open(root+'/Jacobian_null.txt','wb') as f:
        pickle.dump(jacobian_null,f)
    with open(root+'/target_f.txt','wb') as f:
        pickle.dump(target_f,f)
    return sym.lambdify([variables], jacobian_null, "numpy"), sym.lambdify([variables], target_f, "numpy") # Convert a SymPy expression into a function that allows for fast numeric evaluation.

if __name__ == "__main__":
    jacobian_sym_func = jacobian_sym()

    jacobian = sym.Matrix([])
    jacobian_null = sym.Matrix([])
    Hessian = sym.Matrix([])
    
    with open(root+'/Jacobian.txt','rb') as f:
        jacobian = pickle.load(f)
    with open(root+'/target_f.txt','rb') as f:
        target_f = pickle.load(f)
    with open(root+'/Jacobian_null.txt','rb') as f:
        jacobian_null = pickle.load(f)

    q1, q2, q3, q4, q5, q6, q7 = sym.symbols("q_1 q_2 q_3 q_4 q_5 q_6 q_7", real=True) 
    variables = [q1, q2, q3, q4, q5, q6, q7]
    J_func = sym.lambdify([variables], jacobian, modules='numpy')
    Jn_func = sym.lambdify([variables], jacobian_null, modules='numpy')
    H_func = sym.lambdify([variables], target_f, modules='numpy')

    dill.settings['recurse'] = True
    dill.dump(J_func, open(root+'/J_func_simp', "wb"))
    dill.dump(Jn_func, open(root+'/Jn_func_simp', "wb"))
    dill.dump(H_func, open(root+'/H_func_simp', "wb"))