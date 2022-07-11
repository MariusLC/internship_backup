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

import itertools
import random


if __name__ == '__main__':

    # Environnement ethical dimension
    env_dim = 4

    # order of preference on objectives
    order = [3,1,0,2]

    # Preference giver to target
    # preference_giver = EthicalParetoGiverv3()
    preference_giver = EthicalParetoGiverv3_ObjectiveOrder(order)

    

    # preference learner (model) get states and action or traj objectives as input
    statesOrScores = False

    # Create All combinations of possible queries
    # all_combi = [range(13), range(13), range(13), range(13), range(-8,1)]
    # all_combi = [np.array(c) for c in all_combi]
    # list_tuple_combi = list(itertools.product(*all_combi))
    # random.shuffle(list_tuple_combi)

    # action combi
    all_combi_actions = [[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]
    all_combi_actions = [np.array(c) for c in all_combi_actions]
    list_tuple_combi_actions = list(itertools.product(all_combi_actions, repeat=2))
    random.shuffle(list_tuple_combi_actions)


    n_queries = 100
    n_epochs = 2000
    batch_size_loss = 5

    # preference_model_filename = "generated_data/v3/pref_model/1000q_ParetoDom.pt"
    preference_model_filename = "generated_data/v3/pref_model/trajectories/ALLCOMBI_"+str(n_queries)+"q_"+str(batch_size_loss)+"b_"+str(n_epochs)+"e_"+str(order)+".pt"

    # Config
    wandb.init(project='PrefTrain', config={
        'env_id': 'randomized_v3',
        'n_queries': n_queries,
        'batch_size_loss': batch_size_loss,
        'n_epochs' : n_epochs,
        'lr_reward': 3e-5,
        'preference_model_filename':preference_model_filename,
        'n_workers': 12,
        'n_steps':n_epochs*batch_size_loss
    })
    config = wandb.config

    # # Create Environment
    # vec_env = SubprocVecEnv([make_env(config.env_id, i) for i in range(config.n_workers)])
    # states = vec_env.reset()
    # states_tensor = torch.tensor(states).float().to(device)

    # # Fetch Shapes
    # n_actions = vec_env.action_space.n
    # obs_shape = vec_env.observation_space.shape
    # state_shape = obs_shape[:-1]
    # in_channels = obs_shape[-1]

    # Preference model to train 
    if statesOrScores :
        preference_model = PreferenceModelMLP(env_dim).to(device)
        preference_buffer = PreferenceBuffer()
    else :
        preference_model = PreferenceModelTEST(env_dim).to(device)
        preference_buffer = PreferenceBufferTest()
        
    preference_optimizer = torch.optim.Adam(preference_model.parameters(), lr=config.lr_reward)

    for i in range(n_queries):
        ret_a = randomlist = random.sample(range(0, 13), 3) + [random.randint(-8, 0)]
        ret_b = randomlist = random.sample(range(0, 13), 3) + [random.randint(-8, 0)]

        auto_preference = preference_giver.query_pair(ret_a, ret_b)
        preference_buffer.add_preference(ret_a, ret_b, auto_preference)

    for i in range(config.n_epochs):
        preference_loss = update_preference_model(preference_model, preference_buffer, preference_optimizer,
                                                      config.batch_size_loss)
        wandb.log({'Preference Loss': preference_loss}, step=i*config.batch_size_loss)

        for j, combi in enumerate(all_combi_actions):
            evaluation = preference_model.evaluate_action(combi)
            wandb.log({str(combi): evaluation}, step=i*config.batch_size_loss)

    save_data(preference_model, preference_model_filename)