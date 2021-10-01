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

Ktheta = np.diag(np.array([1.72, 1.32, 0.52])) # 3x3 

def external_force(): 
  '''
  External force 
  Output : Fx, Fy 
  shape  :(2x1)
  '''

  F = np.array([[1],[1]])
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
  '''
  Homogeneous Transformation matrix
  Input  : joint angle
  Output : HT matrix
  shape  : (4x4)
  '''

  joint_params = np.asarray(joint_params, dtype=float)
  q1, q2, q3 = joint_params
  TF = np.linalg.multi_dot([ 
          Rz(q1),Tx(a1),                   # Joint 1 to 2
          Rz(q2),Tx(a2),                  # Joint 2 to 3 
          Rz(q3),Tx(a3)                    # Joint 3 to E 
  ])
  print("x,y,z", TF[:2,-1]) # position
  return TF

def jacobian_sym(): 
  '''
  symbol jacobian 
  Output : Symbol Jacobian, Target function 
  shape  : (2x3), (3x1)
  '''
  q1, q2, q3 = sym.symbols("q_1 q_2 q_3", real=True)  
  variables = [q1, q2, q3]

  TF =  Rz_sym(q1) @ Tx_sym(a1) @ \
        Rz_sym(q2) @ Tx_sym(a2) @ \
        Rz_sym(q3) @ Tx_sym(a3) 
  R = TF[:3,:-1]

  jacobian = sym.Matrix([])
  Hessian = sym.Matrix([])

  for var in variables:
    T_d  = sym.diff(TF, var) 
    #print(T_d)
    T    = T_d[0:2, -1] # translation?
    # print(T.shape)
    #R_d  = T_d[0:3, :-1] #Rotation diff
    #R_j  = R_d @ R.T  #Rotation jacobian
    # print("var : ", var)
    # print("R_j", R_j)

    #J = T.row_insert(3, sym.Matrix([R_j[2,1], R_j[0,2], R_j[1,0]])) # [T_d; R_d]
    jacobian = jacobian.col_insert(len(jacobian), T) # 6x1 translation + rotation diff 

  F = external_force()
  Ktheta_inv = np.linalg.inv(Ktheta)
  target_jacobian = jacobian @ Ktheta_inv @ jacobian.T @ F # 2x3 @ 3x3 @ 3x2 @ 2x1 = 2x1
  # [dx, dy, dz, rx, ry, rz].T

  H =0.5*(target_jacobian.row(0)@target_jacobian.row(0) + target_jacobian.row(1)@target_jacobian.row(1))

  for var in variables:
    T_d = sym.diff(H, var)
    Hessian = Hessian.col_insert(len(Hessian), T_d) # 3x1

  return sym.lambdify([variables], jacobian, "numpy"), sym.lambdify([variables], Hessian, "numpy") # Convert a SymPy expression into a function that allows for fast numeric evaluation.

def jacobian(joint_params):

  variables = [*joint_params]
  jacobian = jacobian_sym_func(variables)
  hessian = diff_jacobian_sym_func(variables)

  return jacobian, hessian

def diff_jacobian(joint_params):
  variables = [*joint_params]

  return diff_jacobian_sym_func(variables)

def null_space_method(q0, p_goal, weights=[1,3,1], time_step=0.01, max_iteration=5000, accuracy=0.001):

  assert np.linalg.norm(p_goal[:2]) <= 0.85*np.sum([a1, a2, a3, a4]), "Robot Length constraint violated"
  q_n0 = q0
  p = FK(q_n0)[:2,-1] # position 
  p = np.array([[p[0]], [p[1]] ]) # shape miss match (2,1) to (2,)

  # R = FK(q_n0)[:3] # Rotation matrix
  #rpy = euler_angles(R) # roll pitch yaw
  #p = np.array([[p[0]], [p[1]], [p[2]], [rpy[0]], [rpy[1]], [rpy[2]]])

  t_dot = p_goal[:2] - p[:2]
  e = np.linalg.norm(t_dot)
  Tt = np.block([ np.eye(3)])

  q_n1 = q_n0
  δt = time_step
  q_dot_0 =  [ 4, 9, 1] # [ 0.4, 0.9, 0.1, 0.7, 0.9, 0.5, 0.22]

  i=0

  start_time = time.time()
  test =[]
  test2 = []
  while True:
    if (e < accuracy): 
      break

    p = FK(q_n0)[:2,-1] # position 
    p = np.array([[p[0]], [p[1]] ]) # shape miss match (2,1) to (2,)

    #fk = FK(q_n0)
    #rx, ry, rz = euler_angles(fk[:3,:3])
    #p = np.hstack([fk[:3,-1], [rx, ry, rz] ]) # current position and orientation

    t_dot = p_goal[:2] - p[:2] # (2x1)
    e = np.linalg.norm(t_dot) 
    # w_inv = np.linalg.inv(np.diag(weights)) 
    J, H = jacobian(q_n0)
    print("H", H)
    # Jt = np.dot(Tt, jacobian(q_n0)) 
    # print("Jt: ", Jt)

    c =np.array([0.0001, 0.0001])

    # temp  = J@J.T             # 2x3
    # temp2 = c.T@np.eye(2)     # 2x1
    # temp3 = J@J.T + c.T@np.eye(2) #2x2

    # temp4 = np.linalg.inv((J@J.T + c.T@np.eye(2))) # 2x2
    #print(J.shape, temp0.shape, temp.shape, temp2.shape, temp3.shape, temp4.shape, psd_J.shape)
    psd_J = J.T@ np.linalg.inv((J@J.T + c.T@np.eye(2))) # 3x2  # @ inv(2x3 @ 3x2 )          
    qdot = psd_J @ t_dot - (np.eye(3) - (psd_J @ J))@H.T # 3x2 @ 2x1 -( 3x3 - 3x2@ 2x3)@3x1

    temp = psd_J @ t_dot
    temp2 = (np.eye(3) - (psd_J @ J))
    temp3 = H.T
    temp4 = (np.eye(3) - (psd_J @ J))@H.T

    # print("psd_J                        :", psd_J.shape)
    # print("t_dot                        :", t_dot.shape)
    # print("qdot ", qdot)

    # print("psd_J @ t_dot                : ", temp.shape)
    # print("(np.eye(3) - (psd_J @ J))    : ", temp2.shape)
    # print("H.T                          : ", temp3.shape)
    # print("(np.eye(3) - (psd_J @ J))@H.T: ", temp4.shape)

    q_dot = np.array([qdot[0][0],qdot[1][0],qdot[2][0]]) # shape miss match (2,1) to (2,)
    q_n1 = q_n0 + (δt * q_dot)
    q_n0 = q_n1

    dxyz = dcatersian(q_n0)
    print("Deviation: ", dxyz)
    test.append(dxyz[0])
    test2.append(dxyz[1]) # for plot errors

    i+=1
    if (i > max_iteration):
      print("No convergence")
      break
      
  plt.plot(test)
  plt.plot(test2)
  plt.show()
  end_time = time.time()
  print(f"to {np.round(p_goal,2)} :: time taken {np.round(end_time - start_time, 4)} seconds\n")

  return q_n1

def plot_robot(q_parms):

    #print("q_parm : ",q_parms[0])
    q1, q2, q3 = q_parms[0]
  
    T01 = np.eye(4)
    T12 = Rz(q1) @ Tx(a1)          # Joint 1 to 2
    T23 = Rz(q2) @ Tx(a2)                   # Joint 2 to 3
    T3E = Rz(q3) @ Tx(a3)         # Joint 3 to E

    T02 = T01 @ T12
    T03 = T01 @ T12 @ T23
    T0E = T01 @ T12 @ T23 @ T3E

    x_pos = [T01[0,-1], T02[0,-1], T03[0,-1], T0E[0,-1]]
    y_pos = [T01[1,-1], T02[1,-1], T03[1,-1], T0E[1,-1]]
    z_pos = [T01[2,-1], T02[2,-1], T03[2,-1], T0E[2,-1]]
    print(x_pos[0], y_pos[0], x_pos[1], y_pos[1], x_pos[2], y_pos[2])
    print("q1: ",np.rad2deg(q1), " q2: ", np.rad2deg(q2), " q3: ", np.rad2deg(q3))
    
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
              enumerate(np.round(np.rad2deg([ 0, q1, q2, q3]),0)) ],
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

def get_cnfs_null(method_fun, q0=np.deg2rad([0, 0, 0]), kwargs=dict()):
  # x = np.array([1.1464])
  # y = np.array([0.1999999])
  # #z = np.array([0])
  rob_cnfs = []
  pos = np.array([[1.1464], [0.2]])
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
    a1, a2, a3, a4 = 0.4, 0.4, 0.4, 0.4

    pi = np.pi
    pi_sym = sym.pi

    jacobian_sym_func, diff_jacobian_sym_func = jacobian_sym()
    get_cnfs_null(method_fun=null_space_method, q0=np.deg2rad([0.5,0.5,0.5]))