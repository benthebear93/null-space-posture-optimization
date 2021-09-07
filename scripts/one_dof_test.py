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
        self.BallPath = "C:/Users/UNIST/Desktop/git/null-space-posture-optimization/urdf/ball.urdf"
        self.BallPath2 = "C:/Users/UNIST/Desktop/git/null-space-posture-optimization/urdf/ball2.urdf"
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
        self.robotID = p.loadURDF(self.URDFPath, robotStartPos, robotStartOrn, useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.eefID = 8
        self.joints, self.controlJoints = utils_tx90.setup_tx90(p, self.robotID) #get joints and controllable joints
        #print(self.joints)
        # print(self.joints.id)
        self.joint_init = [deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(0)]
        numJoints = p.getNumJoints(self.robotID)
        for _id in range(numJoints):
            _name = p.getJointInfo(self.robotID, _id)
            print(_name)
        print("numJ : ", numJoints)
        ballStartPos = [0,0,0.0]
        balltStartOrn = p.getQuaternionFromEuler([0,0,0])

        ballId = p.loadURDF(self.BallPath, ballStartPos, balltStartOrn, useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        
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
        ee_ori = np.array(ee_state[1])
        return ee_pos, ee_ori
    
    def get_ee2(self):
        ee_state = p.getLinkState(self.robotID, self.eefID-2)
        ee_pos = np.array(ee_state[0])
        ee_ori = np.array(ee_state[1])
        return ee_pos, ee_ori

    def home_pose(self):
        for i, name in enumerate(self.controlJoints):
            p.resetJointState(self.robotID, self.joints[name].id,targetValue=self.joint_init[i],targetVelocity=0)
        p.stepSimulation()

    def inverse_kinematics(self, position, orientation):
        return p.calculateInverseKinematics(
            bodyIndex=self.robotID,
            endEffectorLinkIndex=self.eefID,
            targetPosition=position,
            targetOrientation=orientation,
        )
    def getEulerFromQuaternion(self, orientation):
        temp = p.getEulerFromQuaternion(orientation)
        euler = np.array([0.,0.,0.])
        euler[0] = temp[0]
        euler[1] = temp[1]
        euler[2] = temp[2]
        return euler

    def getQuaternionFromEuler(self, euler_ang):
        return p.getQuaternionFromEuler(euler_ang)
    
    def ballupdate(self, pos, ori):
        ballStartPos = pos
        balltStartOrn = p.getQuaternionFromEuler([0,0,0])

        ballId = p.loadURDF(self.BallPath, ballStartPos, balltStartOrn, useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    def ballupdate2(self, pos, ori):
        ballStartPos = pos
        balltStartOrn = p.getQuaternionFromEuler([0,0,0])

        ballId = p.loadURDF(self.BallPath2, ballStartPos, balltStartOrn, useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)

def Rx(theta):
  return np.array([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.array([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.array([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])



if __name__=="__main__":
    tx90 = Tx90()
    tx90.loadURDF()
    a = 10
    i = 0
    tx90.fk([0.1338, 0.7443, deg2rad(83.18), deg2rad(44.5), deg2rad(-28.1), deg2rad(-55.54)])
    # tx90.fk([0.0, 0.0, deg2rad(90), 0, 0, 0])
    
    pos, ori = tx90.get_ee()
    pos2, ori2 = tx90.get_ee2()
    print("pos: ", pos, "ori: ", ori)
    print("euler :", tx90.getEulerFromQuaternion(ori))
    tx90.ballupdate(pos,ori)
    tx90.ballupdate2(pos2,ori2)
    # R = np.dot(deg2rad(177.302), np.dot(Ry(deg2rad(88.824)), Rx(deg2rad(177.272))))
    for i in range(10):
        euler = tx90.getEulerFromQuaternion(ori)
        euler[2] = euler[2] - deg2rad(1)
        ori = tx90.getQuaternionFromEuler(euler)
        j_pos = tx90.inverse_kinematics(pos, ori)
        tx90.fk(j_pos)
        print("after pos: ", pos, "after ori: ", ori)
        time.sleep(0.5)
    time.sleep(11000)