from moral.ppo import *
from moral.airl import *
from moral.active_learning import *
from moral.preference_giver import *
from envs.gym_wrapper import *
from utils.evaluate_ppo import *
from utils.save_data import *

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
import yaml
import os


# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def moral_train_n_experts(env, ratio, lambd, env_steps_moral, query_freq, non_eth_norm, eth_norm, generators_filenames, discriminators_filenames, moral_filename, rand_filename, non_eth_expert_filename):

    nb_experts = len(generators_filenames)

    # w_posterior_size = 0
    # if env == "randomized_v3":
    #     w_posterior_size = 3
    # elif env == "randomized_v1":
    #     w_posterior_size = 2

    # Config
    wandb.init(
        project='MORAL',
        config={
            'env_id': env,
            'ratio': ratio,
            #'env_steps': 8e6,
            'env_steps': env_steps_moral,
            'batchsize_ppo': 12,
            # 'batchsize_ppo': 12,
            #'n_queries': 50,
            'n_queries': 2,
            'preference_noise': 0,
            'n_workers': 12,
            'lr_ppo': 3e-4,
            'entropy_reg': 0.25,
            'gamma': 0.999,
            'epsilon': 0.1,
            'ppo_epochs': 5,
            'lambd': lambd,
            'eth_norm': eth_norm,
            'non_eth_norm': non_eth_norm,
            'temperature_mcmc' : 2,
            'volumeVsEUS' : True, # False = Akrour
            'prop_w_mode': "moral", 
            'posterior_mode' : "basic_temperature",
            'prior': "marius",
            },
        reinit=True)
    config = wandb.config
    env_steps = int(config.env_steps/config.n_workers)

    # if c["real_params"] :
    #     query_freq = int(env_steps/(config.n_queries+2))

    # Create Environment
    vec_env = VecEnv(config.env_id, config.n_workers)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    print('Initializing and Normalizing Rewards...')
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)



    # Expert i
    discriminator_list = []
    generator_list = []
    # utop_list = []

    rand_agent = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    non_eth_expert = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    non_eth_expert.load_state_dict(torch.load(non_eth_expert_filename, map_location=torch.device('cpu')))

    for i in range(nb_experts):
        discriminator_list.append(Discriminator(state_shape=state_shape, in_channels=in_channels).to(device))
        discriminator_list[i].load_state_dict(torch.load(discriminators_filenames[i], map_location=torch.device('cpu')))
        generator_list.append(PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device))
        generator_list[i].load_state_dict(torch.load(generators_filenames[i], map_location=torch.device('cpu')))


        args = discriminator_list[i].estimate_normalisation_points(eth_norm, rand_agent, generator_list[i], config.env_id, config.gamma, steps=10000)
        # args = discriminator_list[i].estimate_normalisation_points(eth_norm, rand_agent, generator_list[i], config.env_id, config.gamma, steps=1000) # tests
        
        # nadir_point_traj, nadir_point_action = discriminator_list[i].estimate_nadir_point(rand_agent, config, steps=10000)
        # upper_bound, lower_bound, mean, norm_mean = discriminator_list[i].estimate_utopia_all(generator_list[i], config, steps=10000)
        
        # upper_bound, lower_bound, mean, norm_mean = discriminator_list[i].estimate_utopia_all(generator_list[i], config, steps=1000) # tests
        # print("Upper_bound agent "+str(i)+": "+str(upper_bound))
        # print("Lower_bound agent "+str(i)+": "+str(lower_bound))
        # print("Mean agent "+str(i)+": "+str(mean))
        # print("Normalized Mean agent "+str(i)+": "+str(norm_mean))
        # utop_list.append(discriminator_list[i].estimate_utopia(generator_list[i], config))
        # print(f'Reward Normalization 0: {utop_list[i]}')

        discriminator_list[i].set_eval()


    # for i in range(nb_experts):
    #     mean_traj(generator_list[i], discriminator_list, config, eth_norm, steps=10000)


    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)
    dataset.estimate_normalisation_points(non_eth_norm, non_eth_expert, config.env_id, steps=10000)
    # dataset.estimate_normalisation_points(non_eth_norm, non_eth_expert, config.env_id, steps=1000) # tests
    # dataset.estimate_utopia_point(non_eth_expert, config, steps=10000)

    # Active Learning
    preference_learner = PreferenceLearner(d=len(lambd)+1, n_iter=10000, warmup=1000, temperature=config.temperature_mcmc, prior=config.prior)
    # preference_learner = PreferenceLearner(d=len(lambd)+1, n_iter=10000, warmup=1000)
    # preference_learner = PreferenceLearner(d=len(lambd)+1, n_iter=1000, warmup=100) # tests
    w_posterior = preference_learner.sample_w_prior(preference_learner.n_iter)
    w_posterior_mean = w_posterior.mean(axis=0)

    # Log weight vector
    for i in range(len(w_posterior_mean)):
        wandb.log({'w_posterior_mean ['+str(i)+']': w_posterior_mean[i]}, step=0)

    volume_buffer = VolumeBuffer(len(ratio))
    preference_giver = PreferenceGiverv3(ratio=config.ratio)
    # preference_giver = ParetoDominationPreferenceGiverv3(ratio=config.ratio)

    for t in tqdm(range(env_steps)):
        # print("T = ",t)
        # print("query_freq = ", query_freq)

        # Query User
        if t % query_freq == 0 and t > 0:
            best_delta = volume_buffer.best_delta

            # Using ground truth returns for preference elicitation
            res = volume_buffer.best_returns
            # print(res)
            ret_a, ret_b = volume_buffer.best_returns
            # print(f'Found trajectory pair: {(ret_a, ret_b)}')
            # print(f'Corresponding best delta: {best_delta}')
            preference = preference_giver.query_pair(ret_a, ret_b)
            print(f'obtained preference: {preference}')

            # Run MCMC
            preference_learner.log_preference(best_delta, preference)
            preference_learner.log_returns(ret_a, ret_b)
            # w_posterior = preference_learner.mcmc_vanilla(w_posterior_mean)
            w_posterior = preference_learner.mcmc_test(w_posterior_mean, prop_w_mode=config.prop_w_mode, posterior_mode=config.posterior_mode)
            # print("w_posterior = ", w_posterior)
            w_posterior_mean = w_posterior.mean(axis=0)
            print("w_posterior_mean pre norm = ", w_posterior_mean)

            if sum(w_posterior_mean) != 0: 

                # making a 1 norm vector from w_posterior
                w_posterior_mean = w_posterior_mean/np.linalg.norm(w_posterior_mean)

                # # normalize the vector 
                # w_posterior_mean = w_posterior_mean/np.sum(w_posterior_mean)
                
                print(f'New Posterior Mean {w_posterior_mean}')
            else :
                print(f'Keep the current Posterior Mean {w_posterior_mean}')
            print("\n")

            # Log weight vector
            for i in range(len(w_posterior_mean)):
                wandb.log({'w_posterior_mean ['+str(i)+']': w_posterior_mean[i]}, step=t*config.n_workers)

            volume_buffer.reset()

        # Environment interaction
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = vec_env.step(actions)

        # Fetch AIRL rewards
        airl_state = torch.tensor(states).to(device).float()
        airl_next_state = torch.tensor(next_states).to(device).float()


        airl_rewards_list = []
        for j in range(nb_experts):
            # print("expert n = ", j)
            # airl_rewards_list.append(discriminator_list[j].forward(airl_state, airl_next_state, config.gamma).squeeze(1))
            # airl_rewards_list.append(discriminator_list[j].forward_v2(airl_state, airl_next_state, config.gamma).squeeze(1))
            airl_rewards_list.append(discriminator_list[j].forward(airl_state, airl_next_state, config.gamma, eth_norm).squeeze(1))

        for j in range(nb_experts):
            airl_rewards_list[j] = airl_rewards_list[j].detach().cpu().numpy() * [0 if i else 1 for i in done]

        airl_rewards_array = np.array(airl_rewards_list)
        new_airl_rewards = [airl_rewards_array[:,i] for i in range(len(airl_rewards_list[0]))]
        train_ready = dataset.write_tuple_norm(states, actions, None, rewards, new_airl_rewards, done, log_probs)
        # print("train_ready = ", train_ready)
        # print(str(len(dataset.trajectories)) + " < " + str(dataset.batch_size))


        if train_ready:

            # log objective rewards into volume_buffer before normalizing it
            volume_buffer.log_statistics_sum(dataset.log_returns_sum())
            mean_vectorized_rewards = dataset.compute_scalarized_rewards(w_posterior_mean, non_eth_norm, wandb)
            volume_buffer.log_rewards_sum(dataset.log_vectorized_rew_sum())

            # mean_traj(ppo, discriminator_list, config, eth_norm, steps=1000)

            # Log mean vectorized rewards
            for i, vec in enumerate(mean_vectorized_rewards):
                wandb.log({'vectorized_rew_mean ['+str(i)+']': vec}, step=t*config.n_workers)
                wandb.log({'weighted_rew_mean ['+str(i)+']': w_posterior_mean[i] * vec}, step=t*config.n_workers)
            
            # Log Objectives
            obj_ret = np.array(volume_buffer.objective_logs_sum)
            obj_ret_logs = np.mean(obj_ret, axis=0)
            for i, ret in enumerate(obj_ret_logs):
                wandb.log({'Obj_' + str(i): ret}, step=t*config.n_workers)

            # Log total weighted sum
            # for i, ret in enumerate(dataset.log_returns()):
            #     wandb.log({'Returns': ret}, step=t+i)
            wandb.log({'Returns mean': np.mean(dataset.log_rewards())}, step=t*config.n_workers)


            # Update Models
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs, config.entropy_reg)
            #update_policy_v3(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs, config.entropy_reg, wandb)

            if config.volumeVsEUS:
                # Sample two random trajectories & compare expected volume removal with best pair
                if volume_buffer.auto_pref:
                    # new_returns_a, new_returns_b, logs_a, logs_b = volume_buffer.sample_return_pair()
                    new_returns_a, new_returns_b, logs_a, logs_b = volume_buffer.sample_return_pair_v2()
                    volume_buffer.compare_delta(w_posterior, new_returns_a, new_returns_b, logs_a, logs_b, random=False)
                else:
                    # new_returns_a, new_returns_b = volume_buffer.sample_return_pair()
                    new_returns_a, new_returns_b = volume_buffer.sample_return_pair_v2()
                    volume_buffer.compare_delta(w_posterior, new_returns_a, new_returns_b)
            else:
                # Akrour query selection
                volume_buffer.compare_EUS(w_posterior, w_posterior_mean, preference_learner)


            # Reset PPO buffer
            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)
        # save_data(ppo, moral_filename)

    save_data(ppo, moral_filename)