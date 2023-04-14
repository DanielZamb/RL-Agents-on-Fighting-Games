import diambra.arena
import os
import yaml
import numpy as np

from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule

from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":

    # Loading the parameters from the yaml file
    cfg_file = r"C:\Users\USER\Desktop\UNI\TESIS\RL-Agents-on-Fighting-Games\THESIS_AGENTS\PPO_AGENTS\PARALLEL\ppo_cfg_PARALLEL_EVAL.yaml"
    with open(cfg_file, "r") as yaml_file:
        params = yaml.load(yaml_file, Loader=yaml.FullLoader)

    settings = params["settings"]
    wrappers_settings = params["wrappers_settings"]
    time_dep_seed = 42  # You can use any fixed or time-dependent seed value

    # Creating the environment
    env, num_envs = make_sb3_env(
        params["settings"]["game_id"], settings, wrappers_settings, seed=time_dep_seed)

    log_dir = r".\ppo_parallel_tensorboard_logs"

    os.makedirs(log_dir, exist_ok=True)

    print("Activated {} environment(s)".format(num_envs))

    print("Observation space =", env.observation_space)
    print("Act_space =", env.action_space)

    # Policy param
    policy_kwargs = params["policy_kwargs"]

    # PPO settings
    ppo_settings = params["ppo_settings"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]

    learning_rate = linear_schedule(
        ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    clip_range = linear_schedule(
        ppo_settings["clip_range"][0], ppo_settings["clip_range"][1])
    clip_range_vf = clip_range
    batch_size = ppo_settings["batch_size"]
    n_epochs = ppo_settings["n_epochs"]
    n_steps = ppo_settings["n_steps"]

    lastone = r"C:\Users\USER\Desktop\UNI\TESIS\RL-Agents-on-Fighting-Games\THESIS_AGENTS\PPO_AGENTS\PARALLEL\ppo_parallel_tensorboard_logs\check_point_1.5M\autosave_\1000000.zip"

    agent = PPO.load(lastone, env=env,
                     gamma=gamma, learning_rate=learning_rate, clip_range=clip_range,
                     clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                     tensorboard_log=log_dir)

    print(f"Loaded model from {lastone}")

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    # mean_reward, std_reward = evaluate_policy(
    #     agent, agent.get_env(), n_eval_episodes=10)
    # print("Reward: {} (avg) ± {} (std)".format(mean_reward, std_reward))

    # Run trained agent
    observation = env.reset()
    cumulative_reward = 0
    while True:
        env.render()

        action, _state = agent.predict(observation, deterministic=True)

        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if (reward != 0):
            print("Cumulative reward =", cumulative_reward)

        if done:
            observation = env.reset()
            break

    env.close()
