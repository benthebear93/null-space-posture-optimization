
import time
import numpy as np
import sympy as sym
sym.init_printing()
np.set_printoptions(precision=4, suppress=True, linewidth=200)

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