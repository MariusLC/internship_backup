from tqdm import tqdm
from moral.ppo import PPO
import torch
from envs.gym_wrapper import GymWrapper
from envs.randomized_v2 import MAX_STEPS as max_steps_v2
from envs.randomized_v3 import MAX_STEPS as max_steps_v3
import pickle
import argparse
from utils.save_data import *

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_demos_n_experts(env, nb_demos, experts_filenames, demos_filenames):
    for i in range(len(experts_filenames)):
        generate_demos_1_expert(env, nb_demos, experts_filenames[i], demos_filenames[i],)



def generate_demos_1_expert(env_id, nb_demos, expert_filename, demos_filename):

    max_steps = 0
    if env_id == 'randomized_v2':
        max_steps = max_steps_v2
    else :
        max_steps = max_steps_v3

    # Initialize Environment
    env = GymWrapper(env_id)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)
    dataset = []
    episode = {'states': [], 'actions': []}
    episode_cnt = 0

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Load Pretrained PPO
    ppo = PPO(state_shape=state_shape, n_actions=n_actions, in_channels=in_channels).to(device)
    ppo.load_state_dict(torch.load(expert_filename, map_location=torch.device('cpu')))


    for t in tqdm(range((max_steps-1)*nb_demos)):
        actions, log_probs = ppo.act(states_tensor)
        next_states, reward, done, info = env.step(actions)
        episode['states'].append(states)
        # Note: Actions currently append as arrays and not integers!
        episode['actions'].append(actions)

        if done:
            next_states = env.reset()
            dataset.append(episode)
            episode = {'states': [], 'actions': []}

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    # pickle.dump(dataset, open(demos_filename, 'wb'))
    save_demos(dataset, demos_filename)
