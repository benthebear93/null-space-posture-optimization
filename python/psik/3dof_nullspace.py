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

# euler angle from rotation matrix
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
  q1, q2, q3 = joint_params
  TF = np.linalg.multi_dot([ 
          Rz(q1),Tx(a1),                   # Joint 1 to 2
          Tx(q2),Tx(a2),                  # Joint 2 to 3 
          Rx(q3),Tx(a3)                    # Joint 3 to E 
  ])

  return TF

def plot_robot(q_parms):

    q1, q2, q3 = q_parms
  
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
    a1, a2, a3, a4 = 0.36, 0.42, 0.4, 0.126

    pi = np.pi
    pi_sym = sym.pi

    P = np.sin(np.linspace(-2.5,2.5))
    #jacobian_sym_func = jacobian_sym()
    plot_robot(np.deg2rad([45,0,0]))
    # #get_cnfs(method_fun=null_space_method, q0=np.deg2rad([0,30,0,-20,0,45,0]))
    # get_cnfs_priority(method_fun=null_space_method, q0=np.deg2rad([0,30,0,-20,0,45]))