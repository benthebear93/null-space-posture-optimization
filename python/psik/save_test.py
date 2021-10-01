import numpy as np
import sympy as sym
import pickle
from twist import *

r = 10
p = 10
T = sym.Matrix([ [1, 0, sym.sin(p)],
                    [0, sym.cos(r), -sym.cos(p)*sym.sin(r)],
                    [0, sym.sin(r), sym.cos(p)*sym.cos(r)] ])

T = T.inv()
print(T)