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

def moral_train_n_experts(c, query_freq, env_steps, generators_filenames, discriminators_filenames, moral_filename, non_eth_expert_filename):

    nb_experts = len(generators_filenames)

    # Config
    wandb.init(
        project='MORAL',
        config=c,
        reinit=True)
    config = wandb.config

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
        if config.test:
            args = discriminator_list[i].estimate_normalisation_points(config.eth_norm, rand_agent, generator_list[i], config.env_id, config.gamma, steps=1000) # tests
        else:
            args = discriminator_list[i].estimate_normalisation_points(config.eth_norm, rand_agent, generator_list[i], config.env_id, config.gamma, steps=10000)
        
        discriminator_list[i].set_eval()


    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)
    if config.test:
        dataset.estimate_normalisation_points(config.non_eth_norm, non_eth_expert, config.env_id, steps=1000) # tests
    else :
        dataset.estimate_normalisation_points(config.non_eth_norm, non_eth_expert, config.env_id, steps=10000)

    # Active Learning
    if config.test:
        preference_learner = PreferenceLearner(d=len(config.experts_weights)+1, n_iter=1000, warmup=100, temperature=config.temperature_mcmc, cov_range=config.cov_range, prior=config.prior) # tests
    else :
        preference_learner = PreferenceLearner(d=len(config.experts_weights)+1, n_iter=10000, warmup=1000, temperature=config.temperature_mcmc, cov_range=config.cov_range, prior=config.prior)
    w_posterior = preference_learner.sample_w_prior(preference_learner.n_iter)
    w_posterior_mean = w_posterior.mean(axis=0)

    # Log weight vector
    for i in range(len(w_posterior_mean)):
        wandb.log({'w_posterior_mean ['+str(i)+']': w_posterior_mean[i]}, step=0)

    volume_buffer = VolumeBuffer(len(config.ratio))
    if config.pref_giver_no_null:
        preference_giver = PreferenceGiverv3_no_null(config.ratio)
    else :
        preference_giver = PreferenceGiverv3(config.ratio)

    for t in tqdm(range(env_steps)):

        # Query User
        if t % query_freq == 0 and t > 0:
            ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()
            # best_delta = volume_buffer.best_delta
            best_delta = observed_rew_a-observed_rew_b

            # Using ground truth returns for preference elicitation
            # print(f'Found trajectory pair: {(ret_a, ret_b)}')
            # print(f'Corresponding best delta: {best_delta}')
            preference = preference_giver.query_pair(ret_a, ret_b)
            print(f'obtained preference: {preference}')

            # Run MCMC
            preference_learner.log_preference(best_delta, preference)
            preference_learner.log_returns(observed_rew_a, observed_rew_b)
            w_posterior = preference_learner.mcmc_test(w_posterior_mean, prop_w_mode=config.prop_w_mode, posterior_mode=config.posterior_mode, step=t*config.n_workers)
            w_posterior_mean = w_posterior.mean(axis=0)
            print("w_posterior_mean pre norm = ", w_posterior_mean)
            if sum(w_posterior_mean) != 0: 
                w_posterior_mean = w_posterior_mean/np.linalg.norm(w_posterior_mean)
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
            airl_rewards_list.append(discriminator_list[j].forward(airl_state, airl_next_state, config.gamma, config.eth_norm).squeeze(1).detach().cpu().numpy() * [0 if i else 1 for i in done])

        # for j in range(nb_experts):
        #     airl_rewards_list[j] = airl_rewards_list[j].detach().cpu().numpy() * [0 if i else 1 for i in done]

        airl_rewards_array = np.array(airl_rewards_list)
        new_airl_rewards = [airl_rewards_array[:,i] for i in range(len(airl_rewards_list[0]))]
        train_ready = dataset.write_tuple_norm(states, actions, None, rewards, new_airl_rewards, done, log_probs)

        if train_ready:

            if config.Q_on_actions:
                # save objective rewards into volume_buffer before normalizing it
                volume_buffer.log_statistics_sum(dataset.log_returns_actions())
                objective_logs_sum = dataset.log_returns_sum()
                mean_vectorized_rewards = dataset.compute_scalarized_rewards(w_posterior_mean, config.non_eth_norm, wandb)
                volume_buffer.log_rewards_sum(dataset.log_vectorized_rew_actions())
            else :
                # save objective rewards into volume_buffer before normalizing it
                volume_buffer.log_statistics_sum(dataset.log_returns_sum())
                mean_vectorized_rewards = dataset.compute_scalarized_rewards(w_posterior_mean, config.non_eth_norm, wandb)
                volume_buffer.log_rewards_sum(dataset.log_vectorized_rew_sum())

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
            wandb.log({'Returns mean': np.mean(dataset.log_rewards())}, step=t*config.n_workers)

            # Update Models
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs, config.entropy_reg)

            # rew_a, rew_b, logs_a, logs_b = volume_buffer.sample_return_pair_v2()
            if config.query_selection == "random":
                observed_rew_a, observed_rew_b, ret_a, ret_b = volume_buffer.sample_return_pair_no_batch_reset()
            elif config.query_selection == "random_no_double_null":
                observed_rew_a, observed_rew_b, ret_a, ret_b = volume_buffer.sample_return_pair_no_batch_reset_no_double_zeros()
            elif config.query_selection == "random_less_null":
                observed_rew_a, observed_rew_b, ret_a, ret_b = volume_buffer.sample_return_pair_no_batch_reset_less_zeros_no_double()
            elif config.query_selection == "compare_EUS":
                for k in range(c["nb_query_test"]):
                    volume_buffer.compare_EUS(w_posterior, w_posterior_mean_uniform, preference_learner)
                ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()
            elif config.query_selection == "compare_MORAL":
                for k in range(c["nb_query_test"]):
                    volume_buffer.compare_MORAL(w_posterior)
                ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()
            elif config.query_selection == "compare_basic_log_lik":
                for k in range(c["nb_query_test"]):
                    volume_buffer.compare_delta_basic_log_lik(w_posterior, config.temperature_mcmc)
                ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best() 
            volume_buffer.best_returns = (ret_a, ret_b)
            volume_buffer.best_rewards = (observed_rew_a, observed_rew_b)
            volume_buffer.best_delta = observed_rew_a - observed_rew_b

            # reset buffer ? but not best
            volume_buffer.observed_logs_sum = []
            volume_buffer.objective_logs_sum = []

            # Reset PPO buffer
            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    save_data(ppo, moral_filename)