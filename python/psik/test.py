import numpy as np
import sympy as sym
from twist import *
q1, q2, q3, q4, q5, q6, q7 = sym.symbols("q_1 q_2 q_3 q_4 q_5 q_6 q_7", real=True)  

variables = [q1, q2, q3, q4, q5, q6, q7]

TF =  Rz_sym(q1)@ \
    Rx_sym(q2)              @ \
    Rz_sym(q3) @ \
    Rx_sym(q4)              @ \
    Rz_sym(q5)  @ \
    Rx_sym(q6)              @ \
    Rz_sym(q7)

Tt = np.block([ np.eye(3), np.zeros((3,3)) ])
print(Tt)