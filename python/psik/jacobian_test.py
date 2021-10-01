import time
import sympy as sym
import autograd.numpy as np
sym.init_printing()
from math import atan2, sqrt
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from twist import *

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import autograd.numpy as np
from autograd import grad, jacobian

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

def jacobian_cal(q1,q2,q3,q4,q5,q6):
    variables = [q1, q2, q3, q4, q5, q6]

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


if __name__ == "__main__":
    q1 = 0
    q2 = 0
    q3 = 0
    q4 = 0
    q5 = 0
    q6 = 0
    q = [q1, q2, q3, q4, q5, q6]
    jacobian_cost = jacobian(jacobian_cal, q1)
    print(jacobian_cost)
    #jacobian_cost(np.array([q1, q2, q3, q4, q5, q6]))
