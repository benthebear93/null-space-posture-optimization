from gym.envs.registration import register

# for reward_type in ["sparse", "dense"]:
#     suffix = "Dense" if reward_type == "dense" else ""
#     kwargs = {
#         "reward_type": reward_type,
#     }

#     register(
#         id="Tx90Reach{}-v1".format(suffix),
#         entry_point="tx90_gym.envs:Tx90ReachEnv",
#         kwargs=kwargs,
#         max_episode_steps=50,
#     )

register(
    id="Tx90Reach-v1",
    entry_point="tx90_gym.envs.tx90_tasks:Tx90ReachEnv",
)