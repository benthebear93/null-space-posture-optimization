import pybullet as p
import numpy as np
import pybullet_data
import utils_tx90
import math as m
from time import sleep
import tx90_gym;		from pathlib import Path

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
    print(eefID, ":", ee_pos)
    return ee_pos

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

base = fk(joint_init)
robotStartPos = base
robotStartOrn = p.getQuaternionFromEuler([0,0,0])
ballId =p.loadURDF(BallPath, robotStartPos, robotStartOrn, useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)
a = 0
while True:
    a = a+1
    joint_init = [deg2rad(0), deg2rad(0), deg2rad(a), deg2rad(0), deg2rad(0), deg2rad(0)]
