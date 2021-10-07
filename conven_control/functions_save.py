import time
import numpy as np
import sympy as sym
sym.init_printing()
from math import atan2, sqrt
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from spatialmath import *
import pickle
import os

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import dill

root = os.getcwd()
jacobian = sym.Matrix([])
jacobian_null = sym.Matrix([])
Hessian = sym.Matrix([])

with open(root+'/param_save/Jacobian.txt','rb') as f:
    jacobian = pickle.load(f)
with open(root+'/param_save/hessian.txt','rb') as f:
    Hessian = pickle.load(f)
with open(root+'/param_save/Jacobian_null.txt','rb') as f:
    jacobian_null = pickle.load(f)

q1, q2, q3, q4, q5, q6, q7 = sym.symbols("q_1 q_2 q_3 q_4 q_5 q_6 q_7", real=True) 
variables = [q1, q2, q3, q4, q5, q6, q7]
J_func = sym.lambdify([variables], jacobian, modules='numpy')
Jn_func = sym.lambdify([variables], jacobian_null, modules='numpy')
H_func = sym.lambdify([variables], Hessian, modules='numpy')

dill.settings['recurse'] = True
dill.dump(J_func, open(root+'\param_save\J_func_simp', "wb"))
dill.dump(Jn_func, open(root+'\param_save\Jn_func_simp', "wb"))
dill.dump(H_func, open(root+'\param_save\H_func_simp', "wb"))
