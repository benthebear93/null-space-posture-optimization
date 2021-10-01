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
        self.URDFPath = "C:/Users/UNIST/Desktop/git/null-space-posture-optimization/RL/urdf/tx90.urdf"
        self.BallPath = "C:/Users/UNIST/Desktop/git/null-space-posture-optimization/RL/urdf/ball.urdf"
        self.BallPath2 = "C:/Users/UNIST/Desktop/git/null-space-posture-optimization/RL/urdf/ball2.urdf"
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
        #print("ee_state", ee_state)
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

    def getMatrixFromQuaternion(self, quat):
        return p.getMatrixFromQuaternion(quat)
    
    def ballupdate(self, pos, ori):
        ballStartPos = pos
        balltStartOrn = p.getQuaternionFromEuler([0,0,0])

        ballId = p.loadURDF(self.BallPath, ballStartPos, balltStartOrn, useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    def ballupdate2(self, pos, ori):
        ballStartPos = pos
        balltStartOrn = p.getQuaternionFromEuler([0,0,0])

        ballId = p.loadURDF(self.BallPath2, ballStartPos, balltStartOrn, useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    def getMotorJointStates(self, robot):
        joint_states = p.getJointStates(self.robotID, range(p.getNumJoints(self.robotID)))
        joint_infos = [p.getJointInfo(self.robotID, i) for i in range(p.getNumJoints(self.robotID))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def getjacobian(self):
        mpos, mvel, mtorq = self.getMotorJointStates(self.robotID)
        result = p.getLinkState(self.robotID, self.eefID,computeLinkVelocity=1,computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result

        zero_vec = [0.0] * len(mpos)
        jac_t, jac_r = p.calculateJacobian(self.robotID, self.eefID, com_trn, mpos, zero_vec, zero_vec)
        jac_t = np.array(jac_t)
        jac_r = np.array(jac_r)
        # print(jac_t, type(jac_t))
        # print("---------------------------")
        # print(jac_r, type(jac_r))
        return jac_t, jac_r

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

# euler angle from rotation matrix
def euler_angles(R, sndSol=True):
  rx = m.atan2(R[2,0], R[2,1])
  ry = m.atan2(m.sqrt(R[0,2]**2 + R[1,2]**2), R[2,2])
  rz = m.atan2(R[0,2], -R[1,2])

  return [rx, ry, rz]

def rpy_rotation(R):
    roll  = m.atan2(R[2][1], R[2][2])
    pitch = m.atan2(-R[2][0], m.sqrt(R[2][1]*R[2][1]+R[2][2]*R[2][2]))
    yaw   = m.atan2(R[1][0], R[0][0])
    print("===== rpy =====")
    print(" ")
    print(roll, pitch, yaw)
    print(" ")
    print(np.rad2deg(roll),np.rad2deg(pitch),np.rad2deg(yaw) )

if __name__=="__main__":
    tx90 = Tx90()
    tx90.loadURDF()
    a = 10
    i = 0
    tx90.fk([0.1338, 0.7443, deg2rad(83.18), deg2rad(44.5), deg2rad(-28.1), deg2rad(-55.54)])
    # tx90.fk([0.0, 0.0, deg2rad(90), 0, 0, 0])
    
    pos, ori = tx90.get_ee()
    print("before pos :", pos)
    pos2, ori2 = tx90.get_ee2() # flage
    # print("pos: ", pos)
    # print (" ")
    # print("ori: ", ori)
    print("euler :", np.rad2deg(tx90.getEulerFromQuaternion(ori)))
    # print(" ")

    print("rotation: ", tx90.getMatrixFromQuaternion(ori))
    euler  = tx90.getEulerFromQuaternion(ori)
    R_matrix = Rz(euler[2])@Ry(euler[1])@Rx(euler[0])
    # print("R from euler:", R_matrix)

    rpy_rotation(R_matrix)
    # Rxyz = euler_angles(R_matrix)
    # print("Rxyz :", np.rad2deg(Rxyz[0]), np.rad2deg(Rxyz[1]), np.rad2deg(Rxyz[2]) )
    # print("R from euler:", Rz(Rxyz[2])@Ry(Rxyz[1])@Rx(Rxyz[0]) )
    tx90.ballupdate(pos,ori)
    tx90.ballupdate2(pos2,ori2)
    # R = np.dot(deg2rad(177.302), np.dot(Ry(deg2rad(88.824)), Rx(deg2rad(177.272))))
    #for i in range(10):
    euler = tx90.getEulerFromQuaternion(ori)
    # print("ori: ", ori)
    # print("euler : " , np.rad2deg(euler))
    # print("Rotation matrix: ")
    # print(Rz(euler[2])@Ry(euler[1])@Rx(euler[0]))
    euler[2] = euler[2] + 0.5
    ori = tx90.getQuaternionFromEuler(euler)
    print(" ")
    print(np.rad2deg(euler))
    j_pos = tx90.inverse_kinematics(pos, ori)
    print("pos :", pos)
    tx90.fk(j_pos)
    pos, ori = tx90.get_ee()
    print("after pos :", pos)
    pos2, ori2 = tx90.get_ee2()
    #print(i, ": after pos: ", pos, "after ori: ",  np.rad2deg(ori))
    # print("   ")
    tx90.getjacobian()
    tx90.ballupdate(pos,ori)
    tx90.ballupdate2(pos2,ori2)
        #time.sleep(0.5)
    time.sleep(11000)