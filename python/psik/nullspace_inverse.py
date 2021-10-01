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

def euler_to_rotation(euler):
  R = Rz(euler[2]) @ Ry(euler[1]) @ Rx(euler[0])
  return R
  
def euler_angles(R, sndSol=True):
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
  T6E = Homgm(dh_param6, q6, offset=0)
  TF = T12@T23@T34@T45@T56@T6E


  return TF

def jacobian_sym():
  q1, q2, q3, q4, q5, q6 = sym.symbols("q_1 q_2 q_3", real=True)  

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
  T6E = Homgm_sym(dh_param6, q6, offset=0)
  TF = T12@T23@T34@T45@T56@T6E


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

  return sym.lambdify([variables], jacobian, "numpy") # Convert a SymPy expression into a function that allows for fast numeric evaluation.

def jacobian(joint_params):
    variables = [*joint_params]
    return jacobian_sym_func(variables)

def simple_pseudo(q0, p_goal, time_step=0.01, max_iteration=3000, accuracy=0.001):

  assert np.linalg.norm(p_goal) <= np.sum([a1, a2, a3]), "Robot Length constraint violated"

  q_n0 = q0
  p = FK(q_n0)[:3,-1]
  t_dot = p_goal - p # error of x,y,z,r,p,y?
  e = np.linalg.norm(t_dot)

  Tt = np.block([np.eye(6)])
  q_n1 = q_n0
  δt = time_step
  i = 0
  start_time = time.time()
  while True:
    if e < accuracy:
      print(f"Accuracy of {accuracy} reached")
      break
    
    p =  FK(q_n0)[:6,-1]
    print("q_n0: ", FK(q_n0))
    print("p: ", p[:3] , " p_goal: ", p_goal)
    t_dot = p_goal - p[:3] # position error x, y

    print("t_dot : ", t_dot)
    e = np.linalg.norm(t_dot) # norm of x, y, z
    print(Tt.shape,)
    J_inv = np.linalg.pinv( (Tt @ jacobian(q_n0)) )  # inv jacobian
    q_dot = J_inv @ t_dot
    q_n1 = q_n0 + (δt * q_dot)  
    q_n0 = q_n1
    i+=1
    if (i > max_iteration):
      print("No convergence")
      break
  end_time = time.time()
  print(f"Total time taken {np.round(end_time - start_time, 4)} seconds\n")

  # print(np.mod(q_n1, 2*np.pi))
  return q_n1 # np.mod(q_n1, 2*np.pi) #  element-wise remainder of division

def null_space_method(q0, p_goal, weights=[1,3,1], time_step=0.01, max_iteration=5000, accuracy=0.001):

  assert np.linalg.norm(p_goal[:2]) <= 0.85*np.sum([a1, a2, a3, a4]), "Robot Length constraint violated"
  q_n0 = q0
  p = FK(q_n0)[:2,-1] # position 
  # R = FK(q_n0)[:3] # Rotation matrix

  #rpy = euler_angles(R) # roll pitch yaw
  #p = np.array([p[0], p[1], p[2], rpy[0], rpy[1], rpy[2]])

  t_dot = p_goal[:2] - p[:2]
  H1 = [0, 0, 0]
  e = np.linalg.norm(t_dot)
  Tt = np.block([ np.eye(3)])

  q_n1 = q_n0
  δt = time_step
  q_dot_0 =  [ 4, 9, 1] # [ 0.4, 0.9, 0.1, 0.7, 0.9, 0.5, 0.22]

  i=0

  start_time = time.time()
  joint1 = []
  joint2 = []
  joint3 = []
  while True:
    if (e < accuracy): 
      break

    p = FK(q_n0)[:2,-1] # position 
    #fk = FK(q_n0)
    #rx, ry, rz = euler_angles(fk[:3,:3])
    #p = np.hstack([fk[:3,-1], [rx, ry, rz] ]) # current position and orientation

    t_dot = p_goal[:2] - p[:2] # (2x1)
    e = np.linalg.norm(t_dot) 
    print("error ", e)
    # w_inv = np.linalg.inv(np.diag(weights)) 
    J = jacobian(q_n0)
    # print("jacobian: ", J.shape)
    # Jt = np.dot(Tt, jacobian(q_n0)) 
    # print("Jt: ", Jt)
    c =np.array([0.0001, 0.0001])

    temp0 = J.T
    temp  = J@J.T
    temp2 = c.T@np.eye(2)
    temp3 = J@J.T + c.T@np.eye(2)

    # print("temp3 :", temp3)
    temp4 = np.linalg.inv((J@J.T + c.T@np.eye(2)))

    # print("temp4 :", temp4)
    psd_J = J.T@ np.linalg.inv((J@J.T + c.T@np.eye(2)))

    # print("seudo inv j: ", psd_J)
    # print(J.shape, temp0.shape, temp.shape, temp2.shape, temp3.shape, temp4.shape, psd_J.shape)
    dh = np.array([2*q_n0[0], 2*q_n0[1], 0])
    q_dot = psd_J @ t_dot - (np.eye(3) - (psd_J @ J))@dh.T
    q_n1 = q_n0 + (δt * q_dot)
    joint1.append(q_n1[0])
    joint2.append(q_n1[1])
    joint3.append(q_n1[2])
    q_n0 = q_n1

    i+=1
    if (i > max_iteration):
      print("No convergence")
      break
      
  end_time = time.time()
  print(f"to {np.round(p_goal,2)} :: time taken {np.round(end_time - start_time, 4)} seconds\n")
  plt.plot(joint1)
  plt.plot(joint2)
  plt.plot(joint3)
  plt.show()
  return q_n1

def plot_robot(q_parms):

    #print("q_parm : ",q_parms[0])
    q1, q2, q3 = q_parms #[0]
  
    T01 = np.eye(4)
    T12 = Rz(q1)@Tz(a1)@Tx(a2)                   # Joint 1 to 2
    T23 = Ry(q2)@Ty(a3)@Tz(a4)                   # Joint 2 to 3
    T3E = Ry(q3)@Tz(a5)                         # Joint 3 to 4

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

def plot_robots(rob_cnfs, traj_x, traj_y, traj_z, traj_fun=(lambda x, y: 0.5 + 0.5*np.sin(3*x*y))):
  """
  rob_cnfs: list of robot configurations and plots each conf
  
  traj_fun: function defined in terms of x and y and plots the 3D function 
  representing the desired trajectory
  """

  fig = go.Figure()

  fig.add_scatter3d(
        x=traj_x,
        y=traj_y,
        z=traj_z,
        hoverinfo='none',
        marker=dict( size=0.1 ),
        name = "desired trajectory"
  )
  print("robot conf : ", rob_cnfs)
  for i,q_params in enumerate(rob_cnfs):
    print("============",i ,"=============")
    print("rob conf: ", len(rob_cnfs))
    print("=========================")
    q1, q2, q3 = q_parms
  
    # T01 = np.eye(4)
    # T12 = Rz(q1) @ Tx(a1)          # Joint 1 to 2
    # T23 = Rz(q2) @ Tx(a2)                   # Joint 2 to 3
    # T3E = Rz(q3) @ Tx(a3)         # Joint 3 to E

    T01 = Rz(q1) @ Tx(a1)
    T12 = Rz(q2) @ Tx(a2)
    T23 = Rz(q3) @ Tx(a3) 

    T02 = T01 @ T12
    T03 = T01 @ T12 @ T23
    T0E = T01 @ T12 @ T23 @ T3E
  
    x_pos = [T01[0,-1], T02[0,-1], T03[0,-1], T0E[0,-1]]
    y_pos = [T01[1,-1], T02[1,-1], T03[1,-1], T0E[1,-1]]
    z_pos = [T01[2,-1], T02[2,-1], T03[2,-1], T0E[2,-1]]
    print("q1: ", q1, " q2: ", q2, " q3: ", q3)
    print("x_pos: ", T0E[0,-1], " y_pos: ", T0E[1,-1], " z_pos: ", T0E[2,-1])

    # Setting opacity
    if (i != 0 and i != len(rob_cnfs)-1): 
      opc=0.3
    else: 
      opc=1
    
    # Width
    wd=10

    fig.add_scatter3d(
        x=np.round(x_pos,2),
        y=np.round(y_pos,2),
        z=z_pos,
        line=dict( color='darkblue', width=wd ),
        hoverinfo="text",
        hovertext=[ f"joint {idx}: {q}" 
            for idx,q in 
              enumerate(np.round(np.rad2deg([ 0, q1, q2, q3]),0)) ],
        marker=dict(
            size=wd/2,
            color=["black", "orange", "yellow", "pink", "blue", "goldenrod", "green", "red"],
            # colorscale='Viridis',
        ),
        opacity=opc,
        showlegend=False,
        name = f"conf {i}"
    )


  fig.layout=dict(
    width=1000,
    height=700,
    scene = dict( 
        camera=dict( eye={ 'x':-1.25, 'y':-1.25, 'z':2 } ),
        aspectratio={ 'x':1.25, 'y':1.25, 'z':1 },
        xaxis = dict( nticks=8, range=[np.min(traj_x)-0.5, np.max(traj_x)+0.5] ),
        yaxis = dict( nticks=8, range=[np.min(traj_y)-0.5, np.max(traj_y)+0.5] ),
        zaxis = dict( nticks=8, range=[-0.05, np.max(traj_z)+0.4] ),
        xaxis_title='Robot x-axis',
        yaxis_title='Robot y-axis',
        zaxis_title='Robot z-axis'),
    title=f"Robot plot in different configurations",
    colorscale=dict(diverging="thermal")
  )
  pio.show(fig)

def get_cnfs(method_fun, q0=np.deg2rad([0, 0, 0]), kwargs=dict()):
  x = np.array([0.0])
  y = np.array([0.96569])
  z = np.array([0])
  rob_cnfs = []

  start_time = time.time()
  for (i, j, k) in zip (x, y, z):# k, z
    pos = [i, j, k] # k

    q = method_fun(q0, pos, **kwargs)
    rob_cnfs.append(q)

  end_time = time.time()
  print(f"\n{np.round(end_time-start_time, 1)} seconds : Total time using {method_fun.__name__} \n")
  if kwargs: print(f"\nParameters used: {kwargs}")

  plot_robot(rob_cnfs)

def get_cnfs_null(method_fun, q0=np.deg2rad([0, 0, 0]), kwargs=dict()):
  x = np.array([1.1464])
  y = np.array([0.1999999])
  #z = np.array([0])
  rob_cnfs = []

  start_time = time.time()
  for (i, j) in zip (x, y):# k, z
    pos = [i, j] # k

    q = method_fun(q0, pos, **kwargs)
    rob_cnfs.append(q)

  end_time = time.time()
  print(f"\n{np.round(end_time-start_time, 1)} seconds : Total time using {method_fun.__name__} \n")
  if kwargs: print(f"\nParameters used: {kwargs}")

  plot_robot(rob_cnfs)
if __name__ == "__main__":
    # Length of Links in meters
    a1, a2, a3, a4, a5 = 0.425, 0.05, 0.05, 0.425, 0.1

    pi = np.pi
    pi_sym = sym.pi

    # P = np.sin(np.linspace(-2.5,2.5))
    #jacobian_sym_func = jacobian_sym()
    plot_robot(np.deg2rad([45, 45, 45]))
    # TE = FK(np.deg2rad([45, 45, 45]))
    #get_cnfs_null(method_fun=null_space_method, q0=np.deg2rad([0.5,0.5,0.5]))
    # get_cnfs(method_fun=simple_pseudo, q0=np.deg2rad([0.5, 0.5, 0.5]))
    
    # get_cnfs_priority(method_fun=null_space_method, q0=np.deg2rad([0,30,0,-20,0,45]))