import numpy as np
from gym import utils

from tx90_gym.envs.core import Task
from tx90_gym.utils import distance

class Reach(Task):
	def __init__(
		self,
		sim,
		get_ee_position,
		reward_type="sparse",
		distance_threshold=0.05,
		goal_range=1.0,
	):
		self.sim = sim
		self.reward_type = reward_type
		self.distance_threshold = distance_threshold
		self.get_ee_position = get_ee_position
		self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
		self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
		with self.sim.no_rendering():
			self._create_scene()
			self.sim.place_visualizer(target=[0, 0, 0], distance=1.5, yaw=45, pitch=-30)
	def _create_scene(self):	
		self.sim.create_plane(z_offset=-0.5)
		self.sim.create_table(length=1.5, width=1.5, height=0.5)
		self.sim.create_sphere(
			body_name="target",
			radius=0.02,
			mass=0.0,
			ghost=False,
			position=[0.0, 0.0, 0.0],
			rgba_color=[0.9, 0.1, 0.1, 1.0],
		)
	def get_goal(self):
		return self.goal.copy()

	def get_obs(self):
		return np.array([])  # no tasak-specific observation

	def get_achieved_goal(self):
		ee_position = np.array(self.get_ee_position())
		return ee_position

	def reset(self):
		self.goal = self._sample_goal()
		self.sim.set_base_pose("target", self.goal, [0, 0, 0, 1]) #this is where env generate goal point

	def _sample_goal(self):
		goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
		return goal

	def is_success(self, achieved_goal, desired_goal):
		d = distance(achieved_goal, desired_goal)
		return (d < self.distance_threshold).astype(np.float32)

	def compute_reward(self, achieved_goal, desired_goal, info):
		#print("ag :",achieved_goal, "dg : ",desired_goal)
		d = distance(achieved_goal, desired_goal)
		if self.reward_type == "sparse":
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			return -d