import random
import os
import yaml
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env

cfg_file = r"C:\Users\Usuario\Desktop\UNI\TESIS\THESIS_AGENTS\DUMMIES\dummy_1\dummy_cfg.yaml"
with open(cfg_file) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

settings = params["settings"]
wrappers_settings = params["wrappers_settings"]
time_dep_seed = 42  # You can use any fixed or time-dependent seed value
env, num_envs = make_sb3_env(
    params["settings"]["game_id"], settings, wrappers_settings, seed=time_dep_seed)


class DummyAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation):
        return [random.choice(range(self.action_space.n))]


agent = DummyAgent(env.action_space)

num_episodes = 10
num_steps = 1000

for episode in range(num_episodes):
    obs = env.reset()
    total_reward = 0

    for step in range(num_steps):
        env.render()

        action = agent.choose_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}: Total reward = {total_reward}")

env.close()
