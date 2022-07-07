from envs.gym_wrapper import *
from stable_baselines3.common.vec_env import SubprocVecEnv
from moral.ppo import *
from tqdm import tqdm
import torch
from drlhp.preference_model import *
from moral.preference_giver import *
import wandb
import pickle
import argparse
from utils.save_data import *


from itertools import product
import random


if __name__ == '__main__':

    # preference_model_filename = "generated_data/v3/pref_model/1000q_ParetoDom.pt"
    preference_model_filename = "generated_data/v3/pref_model/5000q_50b_200e_1>0>2>3.pt"

    # Environnement ethical dimension
    env_dim = 4

    # Config
    wandb.init(project='PrefTrain', config={
        'env_id': 'randomized_v3',
        'n_queries': 5000,
        'batch_size_loss': 50,
        'n_epochs' : 200,
        'lr_reward': 3e-5,
    })
    config = wandb.config

    # Preference model to train 
    preference_model = PreferenceModelTEST(env_dim).to(device)
    preference_buffer = PreferenceBufferTest()
    preference_optimizer = torch.optim.Adam(preference_model.parameters(), lr=config.lr_reward)

    # Preference giver to target
    # preference_giver = EthicalParetoGiverv3()
    preference_giver = EthicalParetoGiverv3_1023()

    # all_combi = fct(env_dim)
    # all_combi = np.array(product(range(2), repeat=env_dim))
    # all_combi = product(range(2), repeat=env_dim)
    all_combi = [[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]
    all_combi = [np.array(c) for c in all_combi]

    # Create Environment
    vec_env = SubprocVecEnv([make_env(config.env_id, i) for i in range(12)])
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    for i in range(config.n_queries):
        # ids = random_comparison(all_combi)
        ret_a, ret_b = random.sample(all_combi, 2)
        # print("ret_a = ", ret_a)
        # print("ret_b = ", ret_b)

        auto_preference = preference_giver.query_pair(ret_a, ret_b)
        # print(auto_preference)

        preference_buffer.add_preference(ret_a, ret_b, auto_preference)

    for i in range(config.n_epochs):
        preference_loss = update_preference_model(preference_model, preference_buffer, preference_optimizer,
                                                      config.batch_size_loss)
        wandb.log({'Preference Loss': preference_loss}, step=i)

    save_data(preference_model, preference_model_filename)