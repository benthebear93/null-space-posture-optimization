import time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from twist import *

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

def plot_robot(q_parms):

    q1 = q_parms[0] 
    q2 = q_parms[1]
    q3 = q_parms[2]
    q4 = q_parms[3]
    q5 = q_parms[4]
    q6 = q_parms[5]

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
    T6E = Homgm(dh_param6, q6, offset=0)

    T01 = np.eye(4)
    T02 = T01 @ T12
    T03 = T01 @ T12 @ T23
    T04 = T01 @ T12 @ T23 @ T34
    T05 = T01 @ T12 @ T23 @ T34 @ T45
    T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56
    T0E = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T6E

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