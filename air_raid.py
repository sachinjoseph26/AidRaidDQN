import gym
from collections import deque
from dqn_agent import DQN_Agent
import time
from IPython import display
import matplotlib.pyplot as plt
from AirRaid_DQN.air_raid_utils import blend_images, process_rgb, process_frame

def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env)
    plt.title("Step: %d %s" % (step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

def main():
    env = gym.make('PongDeterministic-v4', render_mode="rgb_array")
    state_size = (105, 80, 1)
    action_size = env.action_space.n
    episodes = 200
    batch_size = 64
    num_steps = 2500
    gamma = 0.985
    agent = DQN_Agent(env,state_size, action_size,episodes,num_steps,batch_size,gamma)

    load_model_name = ""
    if load_model_name != "":
        agent.load(load_model_name)

    ## Visualize state
    observation = env.reset()
    observation = env.step(1)
    for skip in range(2): # skip the start of each game/
        observation = env.step(0)

    # observation = observation[100::]
    processed_observation = process_frame(observation[0])
    show_state(processed_observation[0])
    agent.train()
            
if __name__ == "__main__":
    main()
