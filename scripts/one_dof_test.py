import os
import numpy as np
import math as m
import pybullet as p
import pybullet_data
import utils_tx90
import time
def deg2rad(degree):
	return degree * m.pi/180

class Tx90():
    def __init__(self):
        self.URDFPath = "C:/Users/UNIST/Desktop/git/null-space-posture-optimization/urdf/tx90.urdf"
        print(self.URDFPath)
        self.numofjoint = 6
        #Desktop\git\null-space-posture-optimization\urdf
    def loadURDF(self):
        print("==========loading URDF==========")
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")

        robotStartPos = [0,0,0.0]
        robotStartOrn = p.getQuaternionFromEuler([0,0,0])
        self.robotID = p.loadURDF(self.URDFPath, robotStartPos, robotStartOrn, useFixedBase = True,flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.eefID = 7
        self.joints, self.controlJoints = utils_tx90.setup_tx90(p, self.robotID) #get joints and controllable joints
        self.joint_init = [deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(0)]

    def fk(self, jointpos):
        for i,name in enumerate(self.controlJoints):
            p.resetJointState(self.robotID, self.joints[name].id, targetValue=jointpos[i],targetVelocity=0)
        p.stepSimulation()
        ee_state = p.getLinkState(self.robotID, self.eefID)
        ee_pos = np.array(ee_state[0])
        return ee_pos

    def get_ee(self):
        ee_state = p.getLinkState(self.robotID, self.eefID)
        ee_pos = np.array(ee_state[0])
        return ee_pos
    
    def home_pose(self):
        for i, name in enumerate(self.controlJoints):
            p.resetJointState(self.robotID,self.joints[name].id,targetValue=self.joint_init[i],targetVelocity=0)
        p.stepSimulation()

if __name__=="__main__":
    tx90 = Tx90()
    tx90.loadURDF()
    time.sleep(1000)