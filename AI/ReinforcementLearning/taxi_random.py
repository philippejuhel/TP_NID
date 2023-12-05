import gymnasium as gym
import time
        
# We are using the .env on the end of make to avoid training stopping at 200 iterations, which is the default for the new version of Gym (reference).
# see : https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
env = gym.make("Taxi-v3", render_mode="ansi").env
env.reset()
env.render()


# Début d'un épisode
start = time.time()
epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

terminated = False

while not terminated:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)

    if reward == -10:  # Wrong dropoff/pickup
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    

# fin de l'épisode    
end = time.time()
print(f"Training finished in {end-start} seconds.\n")
print("Timesteps taken: {}".format(epochs))
print("Penalties due to wrong dropoff/pickup: {}".format(penalties))
reponse = input("Affichage de l'animation (y/n)?")

if reponse == 'y':
    # On affiche l'animation
    from time import sleep
    from os import system, name

    # define our clear function
    def clear():
    
        # for windows
        if name == 'nt':
            _ = system('cls')
    
        # for mac and linux(here, os.name is 'posix')
        else:
            _ = system('clear')


    def print_frames(frames):
        for i, frame in enumerate(frames):
            clear()
            print(frame['frame'])
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(.1)
            
    print_frames(frames)
