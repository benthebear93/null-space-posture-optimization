import numpy as np
from gym import spaces
import tx90_gym;		from pathlib import Path
from panda_gym.envs.core import PyBulletRobot
from scipy.spatial.transform import Rotation as R

class Tx90(PyBulletRobot):

	JOINT_INDICES = [0, 1, 2, 3, 4, 5]
	NEUTRAL_JOINT_VALUES = [0.0, 0.0, 1.5707, 0.0, 0.0, 0]
	JOINT_FORCES = [87, 87, 87, 87, 87, 87]
	def __init__(self, sim, base_position = [0, 0, 0]):
		module_path = Path(tx90_gym.__file__)
		self.URDFPath = "%s/urdf/tx90.urdf"%(module_path.parent.parent.absolute())
		
		n_action = 3
		self.action_space = spaces.Box(-1.0, 1.0, shape=(n_action,))
		self.eefID = 8
		super().__init__(
			sim,
			body_name="tx90",
			file_name=self.URDFPath,
			base_position=base_position,
		)
		print("=========================================")
		print("base : ", base_position)
		print("eepos : ", np.array(self.get_ee_position()))
		print("=========================================")

	def set_action(self, action):
		action = action.copy()
		action = np.clip(action, self.action_space.low, self.action_space.high)
		ee_ctrl = action[:3]*0.05
		ee_position = self.get_ee_position()
		target_ee_position = ee_position + ee_ctrl
		# Rx = action[3]
		# Ry = 90
		# Rz = action[4]
		# target_ee_ori = self.euler2quat(Rx, Ry, Rz) 

		 # Clip the height target. For some reason, it has a great impact on learning
		target_ee_position[2] = max(0, target_ee_position[2]) 
		self.euler2quat(0, -90, 0)
		target_angles = self._inverse_kinematics(position=target_ee_position, orientation=[0, 0, 0, 1])#self.euler2quat(0, 90, 0))#[1, 0, 0, 0])#target_ee_ori
		self.control_joints(target_angles=target_angles)

	def get_obs(self):
		# end-effector position and velocity
		ee_position = np.array(self.get_ee_position())
		ee_velocity = np.array(self.get_ee_velocity())
		obs = np.concatenate((ee_position, ee_velocity))

		return obs

	def reset(self):
		self.set_joint_neutral()

	def get_ee_position(self):
		"""Returns the position of the ned-effector as (x, y, z)"""
		return self.get_link_position(self.eefID)

	def get_ee_velocity(self):
		"""Returns the velocity of the end-effector as (vx, vy, vz)"""
		return self.get_link_velocity(self.eefID)  

	def _inverse_kinematics(self, position, orientation):
		"""Compute the inverse kinematics and return the new joint values. The last two
		coordinates (fingers) are [0, 0].

		Args:
		    position (x, y, z): Desired position of the end-effector.
		    orientation (x, y, z, w): Desired orientation of the end-effector.

		Returns:
		    List of joint values.
		"""
		inverse_kinematics = self.sim.inverse_kinematics(
		    self.body_name, self.eefID, position=position, orientation=orientation
		)
		# Replace the fingers coef by [0, 0]
		inverse_kinematics = list(inverse_kinematics[0:6])
	
		return inverse_kinematics

	def set_joint_neutral(self):
		"""Set the robot to its neutral pose."""
		self.set_joint_values(self.NEUTRAL_JOINT_VALUES)

	def set_ee_position(self, position):
		"""Set the end-effector position. Can induce collisions.

		Warning:
			Make sure that the position does not induce collision.

		Args:
			position (x, y, z): Desired position of the gripper.
		"""
		# compute the new joint angles
		angles = self._inverse_kinematics(position=position,  orientation=self.euler2quat(0, 90, 0))
		self.set_joint_values(angles=angles)

	def set_joint_values(self, angles):
		"""Set the joint position of a body. Can induce collisions.

		Args:
		    angles (list): Joint angles.
		"""
		self.sim.set_joint_angles(self.body_name, joints=self.JOINT_INDICES, angles=angles)


	def euler2quat(self, Rx, Ry, Rz):
		rot = R.from_euler('zyx', [Rz, Ry, Rx], degrees=True)

		# Convert to quaternions and print
		rot_quat = rot.as_quat()
		return rot_quat