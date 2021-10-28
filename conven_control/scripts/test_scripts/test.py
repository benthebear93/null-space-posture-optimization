import os
import numpy as np
import sympy as sym
from spatialmath import *
from math import *
ps = os.getcwd()
print ( os.path.abspath(os.path.join(ps, os.pardir)))

rpy1 = np.deg2rad(np.array([-122.4006, 42.9088, 179.8362]))
q1 = quaternion_from_euler(rpy1[0],rpy1[1],rpy1[2])
print("q1", q1)
rpy2 = np.deg2rad(np.array([-57.59, 42.914, 0.1721]))
q2 = quaternion_from_euler(rpy2[0],rpy2[1],rpy2[2])
print("q2", q2)

r1 = rotation_from_euler(rpy1)
r2 = rotation_from_euler(rpy2)

print("r1", r1)
print("r2", r2)
print(" ")
#q1 = [0.17746, 0.815494, -0.44875, 0.3195]
rq1 = quaternion_matrix(q1)
rq2 = quaternion_matrix(q2)
print("rq1", rq1)
print("rq2", rq2)
