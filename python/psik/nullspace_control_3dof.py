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

def Jacobian(l1,l2,l3):
    l1, l2, l3 theta, alpha, a, d = sp.symbols('l1,l2,l3,theta,alpha,a,d')
    rot = sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0],
                     [sp.sin(theta) * sp.cos(alpha), sp.cos(theta) * sp.cos(alpha), -sp.sin(alpha)],
                     [sp.sin(theta) * sp.sin(alpha), sp.cos(theta) * sp.sin(alpha), sp.cos(alpha)]])  

    trans = sp.Matrix([a, -d * sp.sin(alpha), 0]) 
    last_row = sp.Matrix([[0, 0, 0, 1]]) 
    m = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    org_0 = m.subs({alpha: 0, a: l1, theta: theta1, d: 0})  # 0->1 Translation Matrix
    org_1 = m.subs({alpha: 0, a: l2, theta: theta2, d: 0})  # 1->2 Translation Matrix
    org_2 = m.subs({alpha: 0, a: l3, theta: theta3, d: 0})  # 2->3 Translation Matrix
   
    org1 = org_0
    org2 = np.dot(org1,org_1)
    org3 = np.dot(org2,org_2)

    X_L = org3[0,3]
    Y_L = org3[1,3]
    Z_L = org3[2,3]