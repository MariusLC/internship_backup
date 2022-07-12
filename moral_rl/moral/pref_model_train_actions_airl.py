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

from moral.airl import *


if __name__ == '__main__':

    airl_agents_lambda = [[0,1,0,1],[0,0,1,1]]
    eth_norm = "v6"
    non_eth_norm = "v5"

    # Environnement ethical dimension
    env_dim = len(airl_agents_lambda)+1

    # order of preference on objectives
    order = [1,0,2,3]

    # Preference giver to target
    # preference_giver = EthicalParetoGiverv3()
    preference_giver = EthicalParetoGiverv3_ObjectiveOrder(order)

    # preference learner (model) get states and action or traj objectives as input
    statesOrScores = False

    env_steps = 10000
    n_queries = 100
    n_epochs = 2000
    batch_size_loss = 5

    # preference_model_filename = "generated_data/v3/pref_model/1000q_ParetoDom.pt"
    preference_model_filename = "generated_data/v3/pref_model/airl/trajectories/ALLCOMBI_"+str(batch_size_loss)+"b_"+str(n_epochs)+"e_"+str(order)+".pt"

    # Config
    wandb.init(project='PrefTrain', config={
        'env_id': 'randomized_v3',
        'n_queries': n_queries,
        'batch_size_loss': batch_size_loss,
        'n_epochs' : n_epochs,
        'lr_reward': 3e-5,
        'preference_model_filename':preference_model_filename,
        'n_workers': 12,
        'n_steps':n_epochs*batch_size_loss,
        'env_steps': env_steps,
        'gamma':0.999,
        'batchsize_ppo': 12,
    })
    config = wandb.config

    # Create Environment
    vec_env = SubprocVecEnv([make_env(config.env_id, i) for i in range(config.n_workers)])
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    rand_agent = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    non_eth_expert_filename = "generated_data/v3/"+str([1,0,0,0])+"/expert.pt"
    non_eth_expert = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    non_eth_expert.load_state_dict(torch.load(non_eth_expert_filename, map_location=torch.device('cpu')))

    # airl agents
    airl_agents = []
    for i, lmbd in enumerate(airl_agents_lambda):
        airl_agent_filename = "generated_data/v3/"+str(lmbd)+"/discriminator.pt"
        airl_policy_filename = "generated_data/v3/"+str(lmbd)+"/generator.pt"
        airl_agents.append(Discriminator(state_shape=state_shape, in_channels=in_channels).to(device))
        airl_agents[i].load_state_dict(torch.load(airl_agent_filename, map_location=torch.device('cpu')))
        airl_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
        airl_policy.load_state_dict(torch.load(airl_policy_filename, map_location=torch.device('cpu')))
        args = airl_agents[i].estimate_normalisation_points(eth_norm, rand_agent, airl_policy, config.env_id, config.gamma, steps=10000)
        airl_agents[i].set_eval()

    dataset.estimate_normalisation_points(non_eth_norm, non_eth_expert, config.env_id, steps=10000)

    # Preference model to train 
    if statesOrScores :
        preference_model = PreferenceModelMLP(env_dim).to(device)
        preference_buffer = PreferenceBuffer()
    else :
        preference_model = PreferenceModelTEST(env_dim).to(device)
        preference_buffer = PreferenceBufferTest()
        
    preference_optimizer = torch.optim.Adam(preference_model.parameters(), lr=config.lr_reward)

    for t in tqdm(range(int(config.env_steps / config.n_workers))):
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = vec_env.step(actions)

        # Fetch AIRL rewards
        airl_state = torch.tensor(states).to(device).float()
        airl_next_state = torch.tensor(next_states).to(device).float()

        # airl rewards
        airl_rewards = []
        for j, airl_agent in enumerate(airl_agents):
            airl_rewards.append(airl_agent.forward(airl_state, airl_next_state, config.gamma, eth_norm).squeeze(1))

        for j in range(len(airl_agents_lambda)):
            airl_rewards[j] = airl_rewards[j].detach().cpu().numpy() * [0 if i else 1 for i in done]

        airl_rewards = np.array(airl_rewards)
        new_airl_rewards = [airl_rewards[:,i] for i in range(len(airl_rewards[0]))]
        
        dataset.write_tuple_norm(states, actions, None, rewards, new_airl_rewards, done, log_probs)

    # log objective rewards into volume_buffer before normalizing it
    objective_returns = dataset.log_returns_sum()
    mean_vectorized_rewards = dataset.compute_only_vectorized_rewards(non_eth_norm)
    vectorized_rewards = dataset.log_vectorized_rew_sum()

    for i in range(config.n_queries):
        # random or calculate some way to take different actions or trajectories ?
        id_a = random.randint(0, len(vectorized_rewards))
        id_b = random.randint(0, len(vectorized_rewards))
        print("objective_returns[id_a] = ", objective_returns[id_a])
        print("objective_returns[id_b] = ", objective_returns[id_b])
        print("vectorized_rewards[id_a] = ", vectorized_rewards[id_a])
        print("vectorized_rewards[id_b] = ", vectorized_rewards[id_b])
        # On demande à l'expert à partir des rewards objectifs
        auto_preference = preference_giver.query_pair(objective_returns[id_a], objective_returns[id_b])
        # On donne au NN la préférence en fonction des vectorized rewards
        preference_buffer.add_preference(vectorized_rewards[id_a], vectorized_rewards[id_b], auto_preference)
        preference_buffer.add_obj_ret(objective_returns[id_a], objective_returns[id_b])



    # ### ANALYSE LEARNING PROCESS WITH BASIC ACTION REWARDS
    # all_combi = [[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]
    # all_combi = [np.array(c) for c in all_combi]
    # airl_all_combi = [np.concatenate(([c[0]],[a.forward(c) for a in airl_agents])) for c in all_combi]

    for i in range(config.n_epochs):
        preference_loss = update_preference_model(preference_model, preference_buffer, preference_optimizer,
                                                      config.batch_size_loss)
        wandb.log({'Preference Loss': preference_loss}, step=i*config.batch_size_loss)

        # for j in range(10):
        #     combi = preference_buffer.storage[j][0]
        #     obj_ret =  preference_buffer.storage_obj_ret[j][0]
        #     evaluation = preference_model.evaluate_action(combi)
        #     wandb.log({str(obj_ret)+", vec_rew = "+str(combi): evaluation}, step=i*config.batch_size_loss)

    save_data(preference_model, config.preference_model_filename)