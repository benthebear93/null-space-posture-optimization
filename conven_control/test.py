from sympy import *
import numpy as np

q1, q2, q3, q4, q5, q6, q7= symbols("q_1 q_2 q_3 q_4 q_5 q_6 q_7", real=True) 

#a = Matrix([ [q1*q2*0.3, sin(q1)*q3*q5], [q3*cos(q2)*1.8, sin(q1)*cos(q3)*sin(q5)*0.2]])
a = Matrix([ [q1*q2*0.3], [sin(q1)*cos(q3)*sin(q5)*0.2]])
print(a)
print(a.row(1))

c = a.row(1)*a.row(1)
print(c)
t = Matrix([c])
print(t)