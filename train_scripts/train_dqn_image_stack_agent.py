import gymnasium as gym
import flappy_bird_gymnasium
import stable_baselines3 as sb3

import os

os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_VIDEODRIVER"] = "dummy"

log_dir = "./dqn_image_stack_agent_log"

# create the environment
train_env = gym.make("FlappyBird-rgb-v0")
train_env = gym.wrappers.TimeLimit(train_env, max_episode_steps=3000)
train_env = gym.wrappers.resize_observation.ResizeObservation(train_env, (64, 128))
train_env = gym.wrappers.gray_scale_observation.GrayScaleObservation(
    train_env, keep_dim=True
)
train_env = sb3.common.vec_env.DummyVecEnv([lambda: train_env])
train_env = sb3.common.vec_env.VecFrameStack(train_env, n_stack=4)

# create agent
parameters = {
    "learning_rate": 0.0001,
    "buffer_size": 50000,
    "learning_starts": 5000,
    "batch_size": 128,
    "gamma": 0.99,
    "exploration_fraction": 0.9,
    "target_update_interval": 1000,
    "exploration_initial_eps": 0.1,
    "exploration_final_eps": 0.0001,
}
agent = sb3.DQN("CnnPolicy", train_env, tensorboard_log=log_dir, **parameters)

# create evaluation environment
eval_env = gym.make("FlappyBird-rgb-v0")
eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=3000)
eval_env = gym.wrappers.resize_observation.ResizeObservation(eval_env, (64, 128))
eval_env = gym.wrappers.gray_scale_observation.GrayScaleObservation(
    eval_env, keep_dim=True
)
eval_env = sb3.common.vec_env.DummyVecEnv([lambda: eval_env])
eval_env = sb3.common.vec_env.VecFrameStack(eval_env, n_stack=4)
# create  evaluation callback
eval_callback = sb3.common.callbacks.EvalCallback(
    eval_env,
    n_eval_episodes=10,
    eval_freq=5000,
    log_path=log_dir,
    best_model_save_path=log_dir,
    render=False,
)

agent.learn(total_timesteps=5000000, callback=eval_callback)
