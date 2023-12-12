import gymnasium as gym
from common_functions import *

### Start main program

FOZENLAKE_4x4 = "FrozenLake-v1"     
FOZENLAKE_8x8 = "FrozenLake8x8-v1"

env_id = FOZENLAKE_8x8 # Name of the environment, >>>> SELECT IT <<<< !!!

if env_id == FOZENLAKE_4x4:
    # Training parameters
    n_training_episodes = 10000  # Total training episodes
    learning_rate = 0.7          # Learning rate

    # Evaluation parameters
    n_eval_episodes = 100        # Total number of test episodes

    # Environment parameters
    max_steps = 100               # Max steps per episode
    gamma = 0.95                 # Discounting rate
 
    # Exploration parameters
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.05            # Minimum exploration probability
    decay_rate = 0.0005            # Exponential decay rate for exploration prob
    
else:  # For FOZENLAKE_8x8
    # Training parameters
    n_training_episodes = 200000  # Total training episodes
    learning_rate = 0.7          # Learning rate

    # Evaluation parameters
    n_eval_episodes = 100        # Total number of test episodes

    # Environment parameters
    max_steps = 400               # Max steps per episode
    gamma = 0.95                 # Discounting rate
 
    # Exploration parameters
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.001            # Minimum exploration probability
    decay_rate = 0.00005            # Exponential decay rate for exploration prob

# Video
video_fps = 1


# Create the FrozenLake-v1 environment using 4x4 map and non-slippery version and render_mode="rgb_array"
env = gym.make(env_id, is_slippery=False, render_mode="rgb_array")

# We create our environment with gym.make("<name_of_the_environment>")- `is_slippery=False`: The agent always moves in the intended direction due to the non-slippery nature of the frozen lake (deterministic).
print("Observation Space", env.observation_space)
print("Action Space Shape", env.action_space.n)

state_space = env.observation_space.n
print("There are ", state_space, " possible states")

action_space = env.action_space.n
print("There are ", action_space, " possible actions")


Qtable_frozenlake = initialize_q_table(state_space, action_space)

# Train the Q-Learning agen
Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake, learning_rate, gamma)
print(Qtable_frozenlake)

# Evaluate our Q-Learning agent
#mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
#print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

# Record a video
video_path = env_id+".mp4"
record_video(env, Qtable_frozenlake, video_path, video_fps)