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

jacobian = sym.Matrix([])
jacobian_null = sym.Matrix([])
Hessian = sym.Matrix([])

with open('Jacobian.txt','rb') as f:
    jacobian = pickle.load(f)
with open('hessian.txt','rb') as f:
    Hessian = pickle.load(f)
with open('Jacobian_null.txt','rb') as f:
    jacobian_null = pickle.load(f)

q1, q2, q3, q4, q5, q6 = sym.symbols("q_1 q_2 q_3 q_4 q_5 q_6", real=True) 
variables = [q1, q2, q3, q4, q5, q6]
J_func = sym.lambdify([variables], jacobian, modules='numpy')
Jn_func = sym.lambdify([variables], jacobian_null, modules='numpy')
H_func = sym.lambdify([variables], Hessian, modules='numpy')
print("lambdify?")
dill.settings['recurse'] = True
dill.dump(J_func, open("J_func_simp", "wb"))
dill.dump(Jn_func, open("Jn_func_simp", "wb"))
dill.dump(H_func, open("H_func_simp", "wb"))
