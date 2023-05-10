import gymnasium as gym
import flappy_bird_gymnasium
import stable_baselines3 as sb3

log_dir = "./ppo_vector_agent_log"

# create training environment
train_env = gym.make("FlappyBird-v0")
train_env = gym.wrappers.TimeLimit(train_env, max_episode_steps=3000)

# create agent
agent = sb3.ppo.PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-5,
    n_steps=512,
    ent_coef=0.001,
    batch_size=128,
    gae_lambda=0.9,
    n_epochs=20,
    clip_range=0.4,
    policy_kwargs={"log_std_init": -2, "ortho_init": False},
    tensorboard_log=log_dir,
)

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
