import pybullet as p
import numpy as np
import pybullet_data
import utils_tx90
import math as m
from time import sleep
import tx90_gym;		from pathlib import Path

import os
import pandas as pd
import sympy as sym
sym.init_printing()
from pathlib import Path
import matplotlib.pyplot as plt

def _inverse_kinematics(position, orientation):
    inverse_kinematics = sim.inverse_kinematics(
        body_name, eefID, position=position, orientation=orientation
    )
    # Replace the fingers coef by [0, 0]
    inverse_kinematics = list(inverse_kinematics[0:6])

    return inverse_kinematics
    
def deg2rad(degree):
	return degree * m.pi/180

def fk(jointpos):
    for i,name in enumerate(controlJoints):
        p.resetJointState(robotID, joints[name].id, targetValue=jointpos[i],targetVelocity=0)
    p.stepSimulation()
    ee_state = p.getLinkState(robotID, eefID)
    ee_pos = np.array(ee_state[0])
    # print(eefID, ":", ee_pos)
    # print(eefID, ":", ee_state[1])
    return ee_pos

def posture_read(filename):
    root = '/home/benlee/Desktop/git/null-space-posture-optimization/conven_control/data/' #os.getcwd()
    print("reading posture data...")
    df = pd.read_excel(root+filename, header=None, names=None, index_col=None)
    # loc start at 1 , 1~7 for joint values
    num_test = df.loc[1][1:7]
    num_test = df.shape[0]

    # print("number of test: ",  (num_test-1)/2)
    overall_joints =[]
    joint_val = []
    for i in range(1, num_test):
        for j in range(1, 7):
            a = df.iloc[i][j]
            joint_val.append(a)
        print(joint_val)
        overall_joints.append(np.array(joint_val))  
        joint_val = []

    return overall_joints

if __name__ == "__main__":
    module_path = Path(tx90_gym.__file__)
    URDFPath = "%s/urdf/tx90.urdf"%(module_path.parent.parent.absolute())
    BallPath="%s/urdf/tx90.urdf"%(module_path.parent.parent.absolute())
    print("==========loading URDF==========")
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    robotStartPos = [0,0,0.0]
    robotStartOrn = p.getQuaternionFromEuler([0,0,0])

    robotID = p.loadURDF(URDFPath, robotStartPos, robotStartOrn, useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)
    eefID = 8

    joints, controlJoints = utils_tx90.setup_tx90(p, robotID) #get joints and controllable joints
    joint_init = [deg2rad(0), deg2rad(0), deg2rad(90), deg2rad(0), deg2rad(0), deg2rad(0)]
    filename = ['test.xlsx', 'ros_flat_v2.xlsx']
    non_q = posture_read(filename[0])
    opt_q = posture_read(filename[1])
    num_test = len(non_q)
    for i in range(num_test):
        print("test pos :", i)
        base = fk(deg2rad(non_q[i]))
        sleep(1)
        base = fk(deg2rad(opt_q[i]))
        sleep(1)

    # while True:
    #     base = fk(np.deg2rad(np.array([-42.4644,   49.0494,  104.2851, -223.1151,  68.7505,  197.8403])))
    #     sleep(1)
    #     base = fk(np.deg2rad(np.array([ -35.0134 ,  47.5461,  108.424,   167.9444  , 65.3908 ,-175.6493])))
    #     sleep(1)

        # base = fk(np.deg2rad(np.array([-4.4713 ,   52.6902,   77.2336 ,  120.8965, -12.4768, -124.7588])))
        # sleep(1)
        # base = fk(np.deg2rad(np.array([  0.2526 , 56.4472,  72.7631, -77.4118 , 32.9538 , 75.3699])))
        # sleep(3) 

# [   0.7237   -0.0508   -0.2779  179.6617   59.1378 -139.8074] 
#  [  0.7238  -0.0505  -0.278  179.8286  42.9159 122.4065] 

#  ('pose :', x: 0.72566253946
# y: -0.0515362015231
# z: 0.198972564451)

# ('pose :', x: 0.720659108908
# y: -0.0519162667741
# z: 0.202141737481)


