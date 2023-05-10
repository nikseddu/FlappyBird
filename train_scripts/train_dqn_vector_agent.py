import gymnasium as gym
import flappy_bird_gymnasium
import stable_baselines3 as sb3

log_dir = "./dqn_vector_agent_log"

# create training environment
train_env = gym.make("FlappyBird-v0")
train_env = gym.wrappers.TimeLimit(train_env, max_episode_steps=3000)

# create agent
parameters = {
    "learning_rate": 0.0001,
    "buffer_size": 50000,
    "learning_starts": 5000,
    "batch_size":128,
    "gamma": 0.99,
    "exploration_fraction": 0.9,
    "target_update_interval": 1000,
    "exploration_initial_eps": 0.1,
    "exploration_final_eps": 0.0001,
}
agent = sb3.DQN("MlpPolicy", train_env, tensorboard_log=log_dir, **parameters)

# create evaluation environment
eval_env = gym.make("FlappyBird-v0")
eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=3000)
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
