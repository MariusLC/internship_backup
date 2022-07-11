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
    all_combi = [[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]
    all_combi = [np.array(c) for c in all_combi]
    list_tuple_combi = list(itertools.product(all_combi, repeat=2))
    random.shuffle(list_tuple_combi)
    n_queries = len(list_tuple_combi)

    n_epochs = 2000
    batch_size_loss = 5

    # preference_model_filename = "generated_data/v3/pref_model/1000q_ParetoDom.pt"
    preference_model_filename = "generated_data/v3/pref_model/ALLCOMBI_"+str(batch_size_loss)+"b_"+str(n_epochs)+"e_"+str(order)+".pt"

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

    # Preference model to train 
    if statesOrScores :
        preference_model = PreferenceModelMLP(env_dim).to(device)
        preference_buffer = PreferenceBuffer()
    else :
        preference_model = PreferenceModelTEST(env_dim).to(device)
        preference_buffer = PreferenceBufferTest()
        
    preference_optimizer = torch.optim.Adam(preference_model.parameters(), lr=config.lr_reward)

    for i, combi in enumerate(list_tuple_combi):
        auto_preference = preference_giver.query_pair(combi[0], combi[1])
        preference_buffer.add_preference(combi[0], combi[1], auto_preference)

    for i in range(config.n_epochs):
        preference_loss = update_preference_model(preference_model, preference_buffer, preference_optimizer,
                                                      config.batch_size_loss)
        wandb.log({'Preference Loss': preference_loss}, step=i*config.batch_size_loss)

        for j, combi in enumerate(all_combi):
            evaluation = preference_model.evaluate_action(combi)
            wandb.log({str(combi): evaluation}, step=i*config.batch_size_loss)

    save_data(preference_model, preference_model_filename)