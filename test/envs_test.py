import gym
import tx90_gym


def run_env(env):
    """Tests running panda gym envs."""
    done = False
    env.reset()
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
    env.close()


def test_reach():
    env = gym.make("Tx90Reach-v1")
    run_env(env)
