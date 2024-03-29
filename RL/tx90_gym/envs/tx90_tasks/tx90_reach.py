from tx90_gym.envs.core import RobotTaskEnv
from tx90_gym.pybullet import PyBullet
from tx90_gym.envs.robots import Tx90
from tx90_gym.envs.tasks import Reach


class Tx90ReachEnv(RobotTaskEnv):
    """Reach task wih Tx90 robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
    """

    def __init__(self, render=False, reward_type="sparse"):
        self.sim = PyBullet(render=render)
        self.robot = Tx90(self.sim, base_position=[0.0, 0.0, 0.0])
        self.task = Reach(
            self.sim,
            reward_type=reward_type,
            get_ee_position=self.robot.get_ee_position,
        )
        RobotTaskEnv.__init__(self)
