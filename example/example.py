import gym
import tx90_gym


env = gym.make("Tx90Reach-v1", render=True)

obs = env.reset()
for _ in range(50):
    env.render()
    action = env.action_space.sample()
    env.step(action)

env.close()
