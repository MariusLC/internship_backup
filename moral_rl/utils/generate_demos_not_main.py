from tqdm import tqdm
from moral.ppo import PPO
import torch
from envs.gym_wrapper import GymWrapper
from envs.randomized_v2 import MAX_STEPS as max_steps_v2
from envs.randomized_v3 import MAX_STEPS as max_steps_v3
import pickle
import argparse
from utils.save_data import *

from envs.gym_wrapper import *
from moral.ppo import * 
# from moral.moral_train_not_main import *
from moral.airl import *

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_demos_n_experts(env, nb_demos, experts_filenames, demos_filenames):
    for i in range(len(experts_filenames)):
        generate_demos_1_expert(env, nb_demos, experts_filenames[i], demos_filenames[i],)


def generate_demos_1_moral_agent(env_id, nb_demos, n_workers, gamma, expert_filename, demos_filename, non_eth_expert_filename, non_eth_norm, eth_norm, discriminators_filenames, generators_filenames):
    max_steps = 0
    if env_id == 'randomized_v2':
        max_steps = max_steps_v2
    else :
        max_steps = max_steps_v3

    nb_experts = len(discriminators_filenames)

    # Initialize Environment
    # env = GymWrapper(env_id)
    env = VecEnv(env_id, n_workers)
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Load Pretrained PPO
    ppo = PPO(state_shape=state_shape, n_actions=n_actions, in_channels=in_channels).to(device)
    ppo.load_state_dict(torch.load(expert_filename, map_location=torch.device('cpu')))

    # Discriminators
    rand_agent = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    non_eth_expert = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    non_eth_expert.load_state_dict(torch.load(non_eth_expert_filename, map_location=torch.device('cpu')))
    discriminator_list = []
    generator_list = []
    for i in range(nb_experts):
        discriminator_list.append(Discriminator(state_shape=state_shape, in_channels=in_channels).to(device))
        discriminator_list[i].load_state_dict(torch.load(discriminators_filenames[i], map_location=torch.device('cpu')))
        generator_list.append(PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device))
        generator_list[i].load_state_dict(torch.load(generators_filenames[i], map_location=torch.device('cpu')))
        # args = discriminator_list[i].estimate_normalisation_points(eth_norm, rand_agent, generator_list[i], env_id, gamma=0.999, steps=10000)
        args = discriminator_list[i].estimate_normalisation_points(eth_norm, rand_agent, generator_list[i], env_id, gamma=0.999, steps=1000) # tests
        discriminator_list[i].set_eval()

    dataset = TrajectoryDataset(batch_size=nb_demos, n_workers=n_workers)
    # dataset.estimate_normalisation_points(non_eth_norm, non_eth_expert, env_id, steps=10000)
    dataset.estimate_normalisation_points(non_eth_norm, non_eth_expert, env_id, steps=1000)

    # for t in tqdm(range((max_steps-1)*nb_demos)): # while train_ready = false ?
    batch_full = False
    while(not batch_full):
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = env.step(actions)

        # Fetch AIRL rewards
        airl_state = torch.tensor(states).to(device).float()
        airl_next_state = torch.tensor(next_states).to(device).float()

        airl_rewards_list = []
        for j in range(nb_experts):
            airl_rewards_list.append(discriminator_list[j].forward(airl_state, airl_next_state, gamma, eth_norm).squeeze(1))

        for j in range(nb_experts):
            airl_rewards_list[j] = airl_rewards_list[j].detach().cpu().numpy() * [0 if i else 1 for i in done]

        airl_rewards_array = np.array(airl_rewards_list)
        new_airl_rewards = [airl_rewards_array[:,i] for i in range(len(airl_rewards_list[0]))]
        batch_full = dataset.write_tuple_norm(states, actions, None, rewards, new_airl_rewards, done, log_probs)

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    print("save = ", demos_filename)
    dataset.compute_only_vectorized_rewards(non_eth_norm)
    print("save = ", demos_filename)
    save_demos(dataset.trajectories, demos_filename)
    print("save = ", demos_filename)

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
