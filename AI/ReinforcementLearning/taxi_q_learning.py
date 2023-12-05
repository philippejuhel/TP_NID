import numpy as np
import gymnasium as gym
import time
import random
from os import system, name

NB_EPISODES = 100000
        
# define our clear function
def clear():

    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

# We are using the .env on the end of make to avoid training stopping at 200 iterations, which is the default for the new version of Gym (reference).
# see : https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
env = gym.make("Taxi-v3", render_mode="ansi").env
env.reset()
q_table = np.zeros([env.observation_space.n, env.action_space.n])

start = time.time()
"""Training the agent"""

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, NB_EPISODES+1):
    state, _ = env.reset()
    terminated = False
    
    while not terminated:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, terminated, truncated, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        
    if i % 100 == 0:
        clear()
        print(f"Episode: {i}")

end = time.time()
print(f"Training finished in {end-start} seconds.\n")


"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state, _ = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    terminated = False
    
    while not terminated:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")