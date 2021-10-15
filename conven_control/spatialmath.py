
import time
import numpy as np
import sympy as sym
from math import *
sym.init_printing()
np.set_printoptions(precision=4, suppress=True, linewidth=200)

def rad2deg(q):
  return q*180/pi

def deg2rad(q):
  return q*pi/180

def find_T(R):
    rpy = euler_from_rotation(R)
    r = rpy[0]
    b = rpy[1]
    yaw = rpy[2]
    T = np.array([ 
        [1, 0, sin(b)], 
        [0, cos(r), -cos(b)*sin(r)], 
        [0, sin(r), cos(b)*cos(r)]
    ])
    return T

def rotation_from_euler(euler):
  R = Rz(euler[2]) @ Ry(euler[1]) @ Rx(euler[0])
  return R

def euler_from_rotation(R, sndSol=True):
    '''
    Rotation to euler angle
    Input  : Rotation matrix
    Output : Rx, Ry, Rz
    '''

    #rx = atan2(R[2,0], R[2,1])
    rx = atan2(R[2,1], R[2,2])
    ry = atan2(-R[2,0], sqrt(R[2,1]*R[2,1]+R[2,2]*R[2,2]))
    #ry = atan2(sqrt(R[0,2]**2 + R[1,2]**2), R[2,2])
    #rz = atan2(R[0,2], -R[1,2])
    rz = atan2(R[1,0], R[0,0])
    # r = np.rad2deg([rx,ry,rz])
    # print(r)
    return [rx, ry, rz]

def Rx(q):
  T = np.array([[1,         0,          0, 0],
                [0, np.cos(q), -np.sin(q), 0],
                [0, np.sin(q),  np.cos(q), 0],
                [0,         0,          0, 1]], dtype=float)
  return T

def Ry(q):
  T = np.array([[ np.cos(q), 0, np.sin(q), 0],
                [         0, 1,         0, 0],
                [-np.sin(q), 0, np.cos(q), 0],
                [         0, 0,         0, 1]], dtype=float)
  return T

def Rz(q):
  T = np.array([[np.cos(q), -np.sin(q), 0, 0],
                [np.sin(q),  np.cos(q), 0, 0],
                [        0,          0, 1, 0],
                [        0,          0, 0, 1]], dtype=float)
  return T

def d_Rx(q):
  T = np.array([[0,          0,          0, 0],
                [0, -np.sin(q), -np.cos(q), 0],
                [0,  np.cos(q), -np.sin(q), 0],
                [0,          0,          0, 0]], dtype=float)
  return T

def d_Ry(q):
  T = np.array([[-np.sin(q), 0,  np.cos(q), 0],
                [         0, 0,          0, 0],
                [-np.cos(q), 0, -np.sin(q), 0],
                [         0, 0,          0, 0]], dtype=float)
  return T

def d_Rz(q):
  T = np.array([[-np.sin(q), -np.cos(q), 0, 0],
                [ np.cos(q), -np.sin(q), 0, 0],
                [         0,          0, 0, 0],
                [         0,          0, 0, 0]], dtype=float)
  return T

def Rx_sym(q):
  return sym.Matrix(
      [[1, 0, 0, 0],
        [0, sym.cos(q), -sym.sin(q), 0],
        [0, sym.sin(q), sym.cos(q), 0],
        [0, 0, 0, 1]]
  )

def Ry_sym(q):
  return sym.Matrix(
      [[sym.cos(q), 0, sym.sin(q), 0],
        [0, 1, 0, 0],
        [-sym.sin(q), 0, sym.cos(q), 0],
        [0, 0, 0, 1]]
  )

def Rz_sym(q):
  return sym.Matrix(
      [[sym.cos(q), -sym.sin(q), 0, 0],
        [sym.sin(q), sym.cos(q), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
  )


def d_Rx_sym(q):
  return sym.Matrix(
      [[0, 0, 0, 0],
        [0, -sym.sin(q), -sym.cos(q), 0],
        [0, sym.cos(q), -sym.sin(q), 0],
        [0, 0, 0, 0]]
  )

def d_Ry_sym(q):
  return sym.Matrix(
      [[-sym.sin(q), 0, sym.cos(q), 0],
        [0, 0, 0, 0],
        [-sym.cos(q), 0, -sym.sin(q), 0],
        [0, 0, 0, 0]]
  )

def d_Rz_sym(q):
  return sym.Matrix(
      [[-sym.sin(q), -sym.cos(q), 0, 0],
        [sym.cos(q), -sym.sin(q), 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])

def Tx(x):
  T = np.array([[1, 0, 0, x],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=float)
  return T

def Ty(y):
  T = np.array([[1, 0, 0, 0],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=float)
  return T

def Tz(z):
  T = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, z],
                [0, 0, 0, 1]], dtype=float)
  return T

def Tx_sym(s):
  return sym.Matrix(
      [[1, 0, 0, s],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
  )

def Ty_sym(s):
  return sym.Matrix(
      [[1, 0, 0, 0],
        [0, 1, 0, s],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
  )

def Tz_sym(s):
  return sym.Matrix(
      [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, s],
        [0, 0, 0, 1]]
  )

def d_Tx(x):
  T = np.array([[0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]], dtype=float)
  return T

def d_Ty(y):
  T = np.array([[0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]], dtype=float)
  return T

def d_Tz(z):
  T = np.array([[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0]], dtype=float)
  return T

def d_Tx_sym():
  return sym.Matrix(
      [[0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]
  )

def d_Ty_sym():
  return sym.Matrix(
      [[0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]
  )

def d_Tz_sym():
  return sym.Matrix(
      [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]]
  )

def Homgm(dh_param, q, offset=0):
  d = dh_param[0]
  a = dh_param[1]
  alpha = dh_param[2]
  q = q + offset

  T = np.array([[np.cos(q), -np.cos(alpha)*np.sin(q), np.sin(alpha)*np.sin(q), a*np.cos(q)],
                [np.sin(q), np.cos(alpha)*np.cos(q),  -np.sin(alpha)*np.cos(q), a*np.sin(q)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0,0,0,1]], dtype=float)

  return T

def Homgm_sym(dh_param, q, offset=0):
  d = dh_param[0]
  a = dh_param[1]
  alpha = dh_param[2]

  q = q + offset

  T = sym.Matrix([[sym.cos(q), -sym.cos(alpha)*sym.sin(q), sym.sin(alpha)*sym.sin(q), a*sym.cos(q)],
                [sym.sin(q), sym.cos(alpha)*sym.cos(q),  -sym.sin(alpha)*sym.cos(q), a*sym.sin(q)],
                [0, sym.sin(alpha), sym.cos(alpha), d],
                [0,0,0,1]])

  return T

