import drawRobotics as dR
import numpy as np
import sympy as sp
import numpy.linalg as lin
import math
import warnings
from sympy.physics.vector import init_vprinting
init_vprinting(use_latex='mathjax',pretty_print=False)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from sympy.physics.mechanics import dynamicsymbols
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def Jacobian(th1,l_2,th3,th4,l_5):
    # Symbol for DH params
    theta1, L1, L2, L4, l2, l5, theta3, theta4, theta, alpha, a, d = sp.symbols('theta1, L1, L2, L4, l2, l5, theta3, theta4, theta, alpha, a, d')

    # Rotation Matrix
    rot = sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0],
                     [sp.sin(theta) * sp.cos(alpha), sp.cos(theta) * sp.cos(alpha), -sp.sin(alpha)],
                     [sp.sin(theta) * sp.sin(alpha), sp.cos(theta) * sp.sin(alpha), sp.cos(alpha)]])  

    # Trans Matrix
    trans = sp.Matrix([a, -d * sp.sin(alpha), d * sp.cos(alpha)]) 
    # Bottom 
    last_row = sp.Matrix([[0, 0, 0, 1]]) 
    # Homogeneous transformation Matrix
    m = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)  

    org_0 = m.subs({alpha: 0, a: 0, theta: theta1, d: 2 * L1})  # 0->1 Translation Matrix
    org_1 = m.subs({alpha: sp.rad(90), a: L1, theta: sp.rad(90), d: l2 - L1})  # 1->2 Translation Matrix
    org_2 = m.subs({alpha: sp.rad(90), a: 0, theta: sp.rad(90) + theta3, d: L2})  # 2->3 Translation Matrix
    org_3 = m.subs({alpha: sp.rad(90), a: 0, theta: sp.rad(90) + theta4, d: 0})  # 3->4 Translation Matrix
    org_4 = m.subs({alpha: sp.rad(90), a: 0, theta: -sp.rad(90), d: l5 + L4})  # 4->5 Translation Matrix

    org1 = org_0
    org2 = np.dot(org1,org_1)
    org3 = np.dot(org2,org_2)
    org4 = np.dot(org3,org_3)
    org5 = np.dot(org4,org_4)

    # x,y,z position
    X_L = org5[0,3]
    Y_L = org5[1,3]
    Z_L = org5[2,3]

    # Velocity Jacobian matrix
    Jv = np.array([[sp.diff(X_L,theta1),sp.diff(X_L,l2),sp.diff(X_L,theta3),sp.diff(X_L,theta4),sp.diff(X_L,l5)],
                    [sp.diff(Y_L,theta1),sp.diff(Y_L,l2),sp.diff(Y_L,theta3),sp.diff(Y_L,theta4),sp.diff(Y_L,l5)],
                    [sp.diff(Z_L,theta1),sp.diff(Z_L,l2),sp.diff(Z_L,theta3),sp.diff(Z_L,theta4),sp.diff(Z_L,l5)]])

    # Angular Velocity Jacobian matrix
    Jw = np.array(np.hstack([np.vstack(org_1[0:3,2]), np.vstack([0,0,0]), np.vstack(org_3[0:3,2]), np.vstack(org_4[0:3,2]), np.vstack([0,0,0])]))
    # Analytic jacobian
    JMat = np.vstack([Jv, Jw])
    JMat = sp.lambdify((theta1, L1, L2, L4, l2, theta3, theta4, l5), JMat, 'math')
    X_L = sp.lambdify((theta1, L1, L2, L4, l2, theta3, theta4, l5), X_L, 'math')
    Y_L = sp.lambdify((theta1, L1, L2, L4, l2, theta3, theta4, l5), Y_L, 'math')
    Z_L = sp.lambdify((theta1, L1, L2, L4, l2, theta3, theta4, l5), Z_L, 'math')

    theta1 = th1
    l2 = l_2
    theta3 = th3
    theta4 = th4
    l5 = l_5
    L1 = 500
    L2 = 10
    L4 = 400
    J_Mat = np.array(JMat(theta1, L1, L2, L4, l2, theta3, theta4, l5), dtype=float)
    X_L = X_L(theta1, L1, L2, L4, l2, theta3, theta4, l5)
    Y_L = Y_L(theta1, L1, L2, L4, l2, theta3, theta4, l5)
    Z_L = Z_L(theta1, L1, L2, L4, l2, theta3, theta4, l5)
    return float(X_L), float(Y_L), float(Z_L), J_Mat

def calcORGs(th1,l_2,th3,th4,l_5):
    theta_1, L1, L2, L4, len2, len5, theta_3, theta_4, thetap, alphap, ap, dp = sp.symbols('theta_1, L1, L2, L4, len2, len5, theta_3, theta_4, thetap, alphap, ap, dp')
    rot = sp.Matrix([[sp.cos(thetap), -sp.sin(thetap), 0],
                     [sp.sin(thetap) * sp.cos(alphap), sp.cos(thetap) * sp.cos(alphap), -sp.sin(alphap)],
                     [sp.sin(thetap) * sp.sin(alphap), sp.cos(thetap) * sp.sin(alphap), sp.cos(alphap)]])  # Rotation Matrix
    trans = sp.Matrix([ap, -dp * sp.sin(alphap), dp * sp.cos(alphap)])  # Porg Matrix
    last_row = sp.Matrix([[0, 0, 0, 1]])  # Bottom
    n = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)  # Translation Matrix

    org0_temp = n.subs({alphap: 0, ap: 0, thetap: theta_1, dp: 2 * L1})  # 0->1 Translation Matrix
    org1_temp = n.subs({alphap: sp.rad(90), ap: L1, thetap: sp.rad(90), dp: len2 - L1})  # 1->2 Translation Matrix
    org2_temp = n.subs({alphap: sp.rad(90), ap: 0, thetap: sp.rad(90) + theta_3, dp: L2})  # 2->3 Translation Matrix
    org3_temp = n.subs({alphap: sp.rad(90), ap: 0, thetap: sp.rad(90) + theta_4, dp: 0})  # 3->4 Translation Matrix
    org4_temp = n.subs({alphap: sp.rad(90), ap: 0, thetap: -sp.rad(90), dp: len5 + L4})  # 4->5 Translation Matrix
    # T0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T1 = np.array(org0_temp)
    T2 = np.array(np.dot(T1, org1_temp))
    T3 = np.array(np.dot(T2, org2_temp))
    T4 = np.array(np.dot(T3, org3_temp))
    T5 = np.array(np.dot(T4, org4_temp))

    T1 = sp.lambdify((theta_1, L1, L2, L4, len2, theta_3, theta_4, len5), T1, 'math')
    T2 = sp.lambdify((theta_1, L1, L2, L4, len2, theta_3, theta_4, len5), T2, 'math')
    T3 = sp.lambdify((theta_1, L1, L2, L4, len2, theta_3, theta_4, len5), T3, 'math')
    T4 = sp.lambdify((theta_1, L1, L2, L4, len2, theta_3, theta_4, len5), T4, 'math')
    T5 = sp.lambdify((theta_1, L1, L2, L4, len2, theta_3, theta_4, len5), T5, 'math')
    theta_1 = th1
    len2 = l_2
    theta_3 = th3
    theta_4 = th4
    len5 = l_5
    L1 = 500
    L2 = 10
    L4 = 400
    org1 = np.array(T1(theta_1, L1, L2, L4, len2, theta_3, theta_4, len5))
    org2 = np.array(T2(theta_1, L1, L2, L4, len2, theta_3, theta_4, len5))
    org3 = np.array(T3(theta_1, L1, L2, L4, len2, theta_3, theta_4, len5))
    org4 = np.array(T4(theta_1, L1, L2, L4, len2, theta_3, theta_4, len5))
    org5 = np.array(T5(theta_1, L1, L2, L4, len2, theta_3, theta_4, len5))
    return org1, org2, org3, org4, org5

def get_ik_sol(goal,origin_state,iter):
    x_orig, y_orig, z_orig, garb0 = Jacobian(origin_state[0],origin_state[1],origin_state[2],origin_state[3],origin_state[4])
    pos0 = np.array([x_orig, y_orig, z_orig, 0, 0, 0])
    x = np.linspace(x_orig, goal[0], iter)
    y = np.linspace(y_orig, goal[1], iter)
    z = np.linspace(z_orig, goal[2], iter)
    x_traj = [x_orig]
    y_traj = [y_orig]
    z_traj = [z_orig]
    theta1_tr = np.array([0])
    l2_tr = np.array([300])
    theta3_tr = np.array([0])
    theta4_tr = np.array([0])
    l5_tr = np.array([100])
    prev_state = np.vstack(origin_state)
    for i in range(len(x)):
        x_temp, y_temp, z_temp, jaco0 = Jacobian(prev_state[0],prev_state[1],prev_state[2],prev_state[3],prev_state[4])
        goal_pos = np.array([x[i], y[i], z[i], 0, 0, 0])
        x_traj.append(x_temp)
        y_traj.append(y_temp)
        z_traj.append(z_temp)
        pinvjaco = np.array(lin.pinv(jaco0), dtype=float)
        vec_e = goal_pos-pos0
        e = np.vstack(vec_e)
        delta_theta = np.matmul(pinvjaco,e)
        theta1_tr = np.append(theta1_tr,prev_state[0]+delta_theta[0])
        l2_tr = np.append(l2_tr, prev_state[1]+delta_theta[1])
        theta3_tr = np.append(theta3_tr, prev_state[2]+delta_theta[2])
        theta4_tr = np.append(theta4_tr, prev_state[3]+delta_theta[3])
        l5_tr = np.append(l5_tr,prev_state[4]+delta_theta[4])
        prev_state = [prev_state[j] + delta_theta[j][0] for j in range(len(delta_theta))]
        pos0 = [x_temp,y_temp,z_temp,0,0,0]
    return theta1_tr, l2_tr, theta3_tr, theta4_tr, l5_tr

def drawObject(org1,org2,org3,org4,org5):
    org0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    dR.drawVector(ax, org0, org1, arrowstyle='-', lineColor='c', proj=False, lineWidth=10)
    dR.drawVector(ax, org1, org2, arrowstyle='-', lineColor='k', proj=False, lineWidth=8)
    dR.drawVector(ax, org2, org3, arrowstyle='-', lineColor='k', proj=False, lineWidth=6)
    dR.drawVector(ax, org3, org4, arrowstyle='-', lineColor='k', proj=False, lineWidth=4)
    dR.drawVector(ax, org4, org5, arrowstyle='-', lineColor='k', proj=False, lineWidth=2)


    ax.set_xlim([-1000,1000]), ax.set_ylim([-1000,1000]), ax.set_zlim([0,2000])
    ax.set_xlabel('X axis'), ax.set_ylabel('Y axis'), ax.set_zlabel('Z axis')

def update(num,th1,L2Length,L3Angle,L4Angle,L5Length,lines):
    theta1 = th1
    l2 = L2Length
    theta3 = L3Angle
    theta4 = L4Angle
    l5 = L5Length
    i=num
    org1,org2,org3,org4,org5 = calcORGs(theta1[i],l2[i],theta3[i],theta4[i],l5[i])
    ax.cla()
    lines = drawObject(org1,org2,org3,org4,org5)
    return lines

print("Enter Destination : ex) 600 600 1200")
xgoal, ygoal, zgoal = map(float,input().split())
destination = [xgoal,ygoal,zgoal]
origin_state = [0,300,0,0,100]
xorg, yorg, zorg, dump = Jacobian(origin_state[0],origin_state[1],origin_state[2],origin_state[3],origin_state[4])
departure = np.array([xorg,yorg,zorg])
print(departure)
print(destination)
trace_th1, trace_l2, trace_th3, trace_th4, trace_l5 = get_ik_sol(destination,origin_state,20)
print(trace_th1)
print(trace_l2)


th1Init = float(0)
l2Init = float(300)
th3Init = float(0)
th4Init = float(0)
l5Init = float(100)
org1, org2, org3, org4, org5 = calcORGs(th1Init,l2Init,th3Init,th4Init,l5Init)

fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.3)
lines = drawObject(org1,org2,org3,org4,org5)
line_ani = animation.FuncAnimation(fig, update, len(trace_th1), fargs = (trace_th1,trace_l2,trace_th3,trace_th4,trace_l5,lines),interval=100,repeat=False,cache_frame_data=False)
ax.view_init(azim=-150,elev=30)
plt.legend()
plt.show()