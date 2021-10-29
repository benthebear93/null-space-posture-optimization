import time
import numpy as np
import sympy as sym
from math import *

sym.init_printing()
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from scipy.spatial.transform import Rotation as R

_NEXT_AXIS = [1, 2, 0, 1]
_EPS = np.finfo(float).eps * 4.0
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}


def wrpy(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    c1 = cos(q1)
    c2 = cos(q2)
    c3 = cos(q3)
    c4 = cos(q4)
    c5 = cos(q5)
    c6 = cos(q6)
    s1 = sin(q1)
    s2 = sin(q2)
    s3 = sin(q3)
    s4 = sin(q4)
    s5 = sin(q5)
    s6 = sin(q6)
    c23 = cos(q2 + q3)
    s23 = sin(q2 + q3)

    r31 = (-s23) * c4 * s5 + (c23) * c5
    r32 = -((-s23) * c4 * c5 - (c23) * s5) * s6 - (-s23) * s4 * c6
    r21 = ((s1 * c23) * c4 + c1 * s4) * s5 + (s1 * s23) * c5
    return r31, r32, r21


def quaternion_from_euler(ai, aj, ak, axes="sxyz"):
    """Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = cos(ai)
    si = sin(ai)
    cj = cos(aj)
    sj = sin(aj)
    ck = cos(ak)
    sk = sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    quaternion = numpy.empty((4,), dtype=numpy.float64)
    if repetition:
        quaternion[i] = cj * (cs + sc)
        quaternion[j] = sj * (cc + ss)
        quaternion[k] = sj * (cs - sc)
        quaternion[3] = cj * (cc - ss)
    else:
        quaternion[i] = cj * sc - sj * cs
        quaternion[j] = cj * ss + sj * cc
        quaternion[k] = cj * cs - sj * sc
        quaternion[3] = cj * cc + sj * ss
    if parity:
        quaternion[j] *= -1

    return quaternion


def euler_from_matrix(matrix, axes="sxyz"):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = atan2(M[i, j], M[i, k])
            ay = atan2(sy, M[i, i])
            az = atan2(M[j, i], -M[k, i])
        else:
            ax = atan2(-M[j, k], M[j, j])
            ay = atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = atan2(M[k, j], M[k, k])
            ay = atan2(-M[k, i], cy)
            az = atan2(M[j, i], M[i, i])
        else:
            ax = atan2(-M[j, k], M[j, j])
            ay = atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def quaternion_matrix(quaternion):
    """
    Return homogeneous rotation matrix from quaternion.
    R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True
    """
    _EPS = np.finfo(float).eps * 4.0
    print(quaternion)
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def rad2deg(q):
    return q * 180 / pi


def deg2rad(q):
    return q * pi / 180


def find_T(R):
    rpy = euler_from_rotation(R)
    r = rpy[0]
    b = rpy[1]
    yaw = rpy[2]
    T = np.array(
        [[1, 0, sin(b)], [0, cos(r), -cos(b) * sin(r)], [0, sin(r), cos(b) * cos(r)]]
    )
    return T


def rotation_from_euler(euler):
    """
  rotation matrxi from euler
  Input : rx, ry, rz
  Output : Roation matrix Rz@Ry@Rx
  """
    R = Rz(euler[2]) @ Ry(euler[1]) @ Rx(euler[0])
    return R


def euler_from_rotation(R, sndSol=True):
    """
    Rotation to euler angle
    Input  : Rotation matrix
    Output : roll, pitch, yaw
    """
    # rx = atan2(R[2,0], R[2,1])
    rx = atan2(R[2, 1], R[2, 2])
    ry = atan2(-R[2, 0], sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2]))
    # ry = atan2(sqrt(R[0,2]**2 + R[1,2]**2), R[2,2])
    # rz = atan2(R[0,2], -R[1,2])
    rz = atan2(R[1, 0], R[0, 0])
    # r = np.rad2deg([rx,ry,rz])
    # print(r)
    return [rx, ry, rz]


def Rx(q):
    T = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(q), -np.sin(q), 0],
            [0, np.sin(q), np.cos(q), 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    return T


def Ry(q):
    T = np.array(
        [
            [np.cos(q), 0, np.sin(q), 0],
            [0, 1, 0, 0],
            [-np.sin(q), 0, np.cos(q), 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    return T


def Rz(q):
    T = np.array(
        [
            [np.cos(q), -np.sin(q), 0, 0],
            [np.sin(q), np.cos(q), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    return T


def d_Rx(q):
    T = np.array(
        [
            [0, 0, 0, 0],
            [0, -np.sin(q), -np.cos(q), 0],
            [0, np.cos(q), -np.sin(q), 0],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    return T


def d_Ry(q):
    T = np.array(
        [
            [-np.sin(q), 0, np.cos(q), 0],
            [0, 0, 0, 0],
            [-np.cos(q), 0, -np.sin(q), 0],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    return T


def d_Rz(q):
    T = np.array(
        [
            [-np.sin(q), -np.cos(q), 0, 0],
            [np.cos(q), -np.sin(q), 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    return T


def Rx_sym(q):
    return sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, sym.cos(q), -sym.sin(q), 0],
            [0, sym.sin(q), sym.cos(q), 0],
            [0, 0, 0, 1],
        ]
    )


def Ry_sym(q):
    return sym.Matrix(
        [
            [sym.cos(q), 0, sym.sin(q), 0],
            [0, 1, 0, 0],
            [-sym.sin(q), 0, sym.cos(q), 0],
            [0, 0, 0, 1],
        ]
    )


def Rz_sym(q):
    return sym.Matrix(
        [
            [sym.cos(q), -sym.sin(q), 0, 0],
            [sym.sin(q), sym.cos(q), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def d_Rx_sym(q):
    return sym.Matrix(
        [
            [0, 0, 0, 0],
            [0, -sym.sin(q), -sym.cos(q), 0],
            [0, sym.cos(q), -sym.sin(q), 0],
            [0, 0, 0, 0],
        ]
    )


def d_Ry_sym(q):
    return sym.Matrix(
        [
            [-sym.sin(q), 0, sym.cos(q), 0],
            [0, 0, 0, 0],
            [-sym.cos(q), 0, -sym.sin(q), 0],
            [0, 0, 0, 0],
        ]
    )


def d_Rz_sym(q):
    return sym.Matrix(
        [
            [-sym.sin(q), -sym.cos(q), 0, 0],
            [sym.cos(q), -sym.sin(q), 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )


def Tx(x):
    T = np.array([[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    return T


def Ty(y):
    T = np.array([[1, 0, 0, 0], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    return T


def Tz(z):
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)
    return T


def Tx_sym(s):
    return sym.Matrix([[1, 0, 0, s], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def Ty_sym(s):
    return sym.Matrix([[1, 0, 0, 0], [0, 1, 0, s], [0, 0, 1, 0], [0, 0, 0, 1]])


def Tz_sym(s):
    return sym.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, s], [0, 0, 0, 1]])


def d_Tx(x):
    T = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
    return T


def d_Ty(y):
    T = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
    return T


def d_Tz(z):
    T = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float)
    return T


def d_Tx_sym():
    return sym.Matrix([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])


def d_Ty_sym():
    return sym.Matrix([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])


def d_Tz_sym():
    return sym.Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])


def Homgm(dh_param, q, offset=0):
    d = dh_param[0]
    a = dh_param[1]
    alpha = dh_param[2]
    q = q + offset

    T = np.array(
        [
            [
                np.cos(q),
                -np.cos(alpha) * np.sin(q),
                np.sin(alpha) * np.sin(q),
                a * np.cos(q),
            ],
            [
                np.sin(q),
                np.cos(alpha) * np.cos(q),
                -np.sin(alpha) * np.cos(q),
                a * np.sin(q),
            ],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )

    return T


def Homgm_sym(dh_param, q, offset=0):
    d = dh_param[0]
    a = dh_param[1]
    alpha = dh_param[2]

    q = q + offset

    T = sym.Matrix(
        [
            [
                sym.cos(q),
                -sym.cos(alpha) * sym.sin(q),
                sym.sin(alpha) * sym.sin(q),
                a * sym.cos(q),
            ],
            [
                sym.sin(q),
                sym.cos(alpha) * sym.cos(q),
                -sym.sin(alpha) * sym.cos(q),
                a * sym.sin(q),
            ],
            [0, sym.sin(alpha), sym.cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )

    return T
