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

def evaluate_airl_from_batch(traj_test, discriminator_list, gamma, non_eth_norm, eth_norm, non_eth_expert, env_id):
    dataset = TrajectoryDataset(batch_size=len(traj_test), n_workers=1)
    # dataset.estimate_normalisation_points(non_eth_norm, non_eth_expert, env_id, steps=10000)
    dataset.estimate_normalisation_points(non_eth_norm, non_eth_expert, env_id, steps=1000)
    for _, traj in tqdm(enumerate(traj_test)):
        for i in range(len(traj["states"])-1):
            actions = traj["actions"][i]
            states = traj["states"][i]
            next_states = traj["states"][i+1]
            rewards = traj["returns"][i]

            # Fetch AIRL rewards
            airl_state = torch.tensor(states).to(device).float()
            airl_next_state = torch.tensor(next_states).to(device).float()

            airl_rewards_list = []
            for d in discriminator_list:
                airl_rewards_list.append(d.forward(airl_state, airl_next_state, gamma, eth_norm).squeeze(1).detach().cpu().numpy())

            airl_rewards_array = np.array(airl_rewards_list)
            # print("airl_rewards_array = ", airl_rewards_array)
            new_airl_rewards = [airl_rewards_array[:,i] for i in range(len(airl_rewards_list[0]))]


            batch_full = dataset.write_tuple_norm([states], [actions], [None], [rewards], new_airl_rewards, [i==len(traj["states"])-2], [0.0])
        # print("\nnew_airl_rewards = ", np.array(dataset.trajectories[-1]["airl_rewards"]).sum(axis=0))
        # print("len new_airl_rewards = ", len(dataset.trajectories[-1]["airl_rewards"]))
        # print("new action 0 = ", dataset.trajectories[-1]["actions"][0])
        # print("old_airl_rewards = ", np.array(traj["airl_rewards"][:-1]).sum(axis=0))
        # print("len old_airl_rewards = ", len(traj["airl_rewards"]))
        # print("old action 0 = ", traj["actions"][0])

    dataset.compute_only_vectorized_rewards(non_eth_norm)
    return dataset.trajectories

def moral_train_n_experts(c, query_freq, env_steps, generators_filenames, discriminators_filenames, moral_filename, non_eth_expert_filename):

    nb_experts = len(generators_filenames)

    # Config
    wandb.init(
        project='MORAL',
        config=c,
        reinit=True)
    config = wandb.config

    ###############
    # INIT ENV
    ###############
    vec_env = VecEnv(config.env_id, config.n_workers)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    ###############
    # INIT AGENT
    ###############
    print('Initializing and Normalizing Rewards...')
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)

    ###############
    # INIT AIRL DISCRIMINATORS & GENERATORS, EVALUATE NORMALISATION POINTS
    ###############
    discriminator_list = []
    generator_list = []
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

    #############
    # TRAJECTORIES BATCH FOR QUALITY ESTIMATION
    #############
    print(os.listdir(c["batch_path"]))
    batch_demo = []
    for file in os.listdir(c["batch_path"]):
        batch_demo.extend(pickle.load(open(c["batch_path"]+"/"+str(file), 'rb')))
        print(str(file) + " with " + str(len(batch_demo)) + " trajectories")
    print("The batch contains "+str(len(batch_demo))+" trajectories")

    # If len(batch_demo) < 2000 then UB and LB will be to close to each other
    assert len(batch_demo) >= 2000
    batch_demo = evaluate_airl_from_batch(batch_demo, discriminator_list, c["gamma"], c["non_eth_norm"], c["eth_norm"], non_eth_expert, config.env_id)


    ###############
    # INIT DATASETS
    ###############

    # PPO DATASET
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)
    if config.test:
        dataset.estimate_normalisation_points(config.non_eth_norm, non_eth_expert, config.env_id, steps=1000) # tests
    else :
        dataset.estimate_normalisation_points(config.non_eth_norm, non_eth_expert, config.env_id, steps=10000)

    # EVALUATE PREF LEARNING DATASET
    current_policy_demo_batch = TrajectoryDataset(batch_size=2000, n_workers=config.n_workers)
    current_policy_demo_batch.utopia_point_expert = dataset.utopia_point_expert
    current_policy_demo_batch.utopia_point_expert_max_1_traj = dataset.utopia_point_expert_max_1_traj


    ###############
    # INIT PREF LEARNER
    ###############
    if config.test:
        preference_learner = PreferenceLearner(d=len(config.experts_weights)+1, n_iter=1000, warmup=100, temperature=config.temperature_mcmc, cov_range=config.cov_range, prior=config.prior) # tests
    else :
        preference_learner = PreferenceLearner(d=len(config.experts_weights)+1, n_iter=10000, warmup=1000, temperature=config.temperature_mcmc, cov_range=config.cov_range, prior=config.prior)
    w_posterior = preference_learner.sample_w_prior(preference_learner.n_iter)
    w_posterior_mean = w_posterior.mean(axis=0)

    # Log weight vector
    for i in range(len(w_posterior_mean)):
        wandb.log({'w_posterior_mean ['+str(i)+']': w_posterior_mean[i]}, step=0)

    ###############
    # INIT VOLUME BUFFER & PREF GIVER
    ###############
    volume_buffer = VolumeBuffer(len(config.ratio))
    if config.pref_giver_no_null:
        preference_giver = PreferenceGiverv3_no_null(config.ratio)
    else :
        preference_giver = PreferenceGiverv3(config.ratio)

    ###############
    # EVALUATE PARAMS DEMO_BATCH FOR PREFERENCE LEARNING QUALITY EVALUATION
    ###############
    LB_batch, UB_batch, mean_weight_eval_rand_batch, min_weight_eval_rand_batch, max_weight_eval_rand_batch, mean_inv, LB_batch_inv, UB_batch_inv = preference_giver.evaluate_quality_params(config, batch_demo)

    ###############
    # START LEARNING
    ###############
    for t in tqdm(range(env_steps)):

        # Query User
        if t % query_freq == 0 and t > 0:
            ################
            # BEFORE ASKING A QUESTION, EVALUATE PREF LEARNING
            ################ 

            ######
            # CURRENT POLICY TRAJECTORIES
            ######
            # Reset Environment
            states_DB = vec_env.reset()
            states_tensor_DB = torch.tensor(states_DB).float().to(device)
            train_ready_current_policy_demo_batch = False
            while not train_ready_current_policy_demo_batch:

                # Environment interaction
                actions_DB, log_probs_DB = ppo.act(states_tensor_DB)
                next_states_DB, rewards_DB, done_DB, info_DB = vec_env.step(actions_DB)

                # Fetch AIRL rewards
                airl_state_DB = torch.tensor(states_DB).to(device).float()
                airl_next_state_DB = torch.tensor(next_states_DB).to(device).float()

                airl_rewards_list_DB = []
                for j in range(nb_experts):
                    airl_rewards_list_DB.append(discriminator_list[j].forward(airl_state_DB, airl_next_state_DB, config.gamma, config.eth_norm).squeeze(1).detach().cpu().numpy() * [0 if i else 1 for i in done_DB])

                airl_rewards_array_DB = np.array(airl_rewards_list_DB)
                new_airl_rewards_DB = [airl_rewards_array_DB[:,i] for i in range(len(airl_rewards_list_DB[0]))]
                train_ready_current_policy_demo_batch = current_policy_demo_batch.write_tuple_norm(states_DB, actions_DB, None, rewards_DB, new_airl_rewards_DB, done_DB, log_probs_DB)

                # Prepare state input for next time step
                states_DB = next_states_DB.copy()
                states_tensor_DB = torch.tensor(states_DB).float().to(device)

            mean_vectorized_rewards = current_policy_demo_batch.compute_scalarized_rewards(w_posterior_mean, config.non_eth_norm, wandb)
            current_policy_trajectories = current_policy_demo_batch.trajectories

            # QUALITY HEURISTIC = NB INVERSIONS, CURRENT POLICY TRAJECTORIES
            nb_inv = preference_giver.evaluate_weights_inversions(config.n_best, w_posterior_mean, current_policy_trajectories)
            print("nb_inv = ", nb_inv)
            wandb.log({'nb_inv': nb_inv}, step=(i+1)*config.nb_mcmc)
            # SCORE VS RANDOM WEIGHTS
            nb_inv_vs_rand = (nb_inv - LB_batch_inv)/(UB_batch_inv - LB_batch_inv)
            print("nb_inv_vs_rand = ", nb_inv_vs_rand)
            wandb.log({'nb_inv vs rand': nb_inv_vs_rand}, step=(i+1)*config.nb_mcmc)

            # QUALITY HEURISTIC = SUM SCORE, CURRENT POLICY TRAJECTORIES
            LB, UB, mean_weight_eval_rand, min_weight_eval_rand, max_weight_eval_rand = preference_giver.evaluate_quality_params(config, current_policy_trajectories)
            weight_eval = preference_giver.normalized_evaluate_weights(config.n_best, w_posterior_mean, traj_test, LB, UB)
            weight_eval_10, weight_eval_10_norm = preference_giver.evaluate_weights_print(10, w_posterior_mean, traj_test)
            print("weight_eval = ", weight_eval)
            print("UB = ", UB)
            print("LB = ", LB)
            wandb.log({'weight_eval': weight_eval}, step=(i+1)*config.nb_mcmc)
            wandb.log({'weight_eval TOP 10': weight_eval_10}, step=(i+1)*config.nb_mcmc)
            wandb.log({'weight_eval norm TOP 10': weight_eval_10_norm}, step=(i+1)*config.nb_mcmc)
            # SCORE VS RANDOM WEIGHTS
            norm_score_vs_rand = (weight_eval - min_weight_eval_rand) / (max_weight_eval_rand - min_weight_eval_rand)
            print("norm_score_vs_rand = ", norm_score_vs_rand)
            wandb.log({'mean_weight_eval_rand': mean_weight_eval_rand}, step=(i+1)*config.nb_mcmc)
            wandb.log({'min_weight_eval_rand': min_weight_eval_rand}, step=(i+1)*config.nb_mcmc)
            wandb.log({'max_weight_eval_rand': max_weight_eval_rand}, step=(i+1)*config.nb_mcmc)
            wandb.log({'norm_score_vs_rand': norm_score_vs_rand}, step=(i+1)*config.nb_mcmc)

            # Reset PPO buffer
            current_policy_demo_batch.reset_trajectories()


            ######
            # BATCH DEMO
            ######
            # QUALITY HEURISTIC = NB INVERSIONS, BATCH DEMO
            nb_inv = preference_giver.evaluate_weights_inversions(config.n_best, w_posterior_mean, batch_demo)
            print("nb_inv = ", nb_inv)
            wandb.log({'nb_inv': nb_inv}, step=(i+1)*config.nb_mcmc)
            # SCORE VS RANDOM WEIGHTS
            nb_inv_vs_rand = (nb_inv - LB_batch_inv)/(UB_batch_inv - LB_batch_inv)
            print("nb_inv_vs_rand = ", nb_inv_vs_rand)
            wandb.log({'nb_inv vs rand': nb_inv_vs_rand}, step=(i+1)*config.nb_mcmc)


            # QUALITY HEURISTIC = SUM SCORE, BATCH DEMO
            weight_eval_batch_not_norm, weight_eval_batch = preference_giver.normalized_evaluate_weights(config.n_best, w_posterior_mean, batch_demo, LB_batch, UB_batch)
            weight_eval_10_batch, weight_eval_10_norm_batch = preference_giver.evaluate_weights_print(10, w_posterior_mean, batch_demo)
            print("weight_eval_batch = ", weight_eval_batch)
            print("UB_batch = ", UB_batch)
            print("LB_batch = ", LB_batch)
            wandb.log({'weight_eval_batch': weight_eval_batch}, step=(i+1)*config.nb_mcmc)
            wandb.log({'weight_eval_batch TOP 10': weight_eval_10_batch}, step=(i+1)*config.nb_mcmc)
            wandb.log({'weight_eval_batch norm TOP 10': weight_eval_10_norm_batch}, step=(i+1)*config.nb_mcmc)
            # SCORE VS RANDOM WEIGHTS
            norm_score_vs_rand_batch = (weight_eval_batch - min_weight_eval_rand_batch) / (max_weight_eval_rand_batch - min_weight_eval_rand_batch)
            print("norm_score_vs_rand_batch = ", norm_score_vs_rand_batch)
            wandb.log({'mean_weight_eval_rand_batch': mean_weight_eval_rand_batch}, step=(i+1)*config.nb_mcmc)
            wandb.log({'min_weight_eval_rand_batch': min_weight_eval_rand_batch}, step=(i+1)*config.nb_mcmc)
            wandb.log({'max_weight_eval_rand_batch': max_weight_eval_rand_batch}, step=(i+1)*config.nb_mcmc)
            wandb.log({'norm_score_vs_rand_batch': norm_score_vs_rand_batch}, step=(i+1)*config.nb_mcmc)



            #############
            # ASK A QUESTION
            #############
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

            # reset everything to clean start again
            current_policy_demo_batch.reset_trajectories()
            volume_buffer.reset()
            # dataset.reset_trajectories()
            states = vec_env.reset()
            states_tensor = torch.tensor(states).float().to(device)

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
                objective_logs_sum = volume_buffer.objective_logs_sum
                mean_vectorized_rewards = dataset.compute_scalarized_rewards(w_posterior_mean, config.non_eth_norm, wandb)
                volume_buffer.log_rewards_sum(dataset.log_vectorized_rew_sum())

            # Log mean vectorized rewards
            for i, vec in enumerate(mean_vectorized_rewards):
                wandb.log({'vectorized_rew_mean ['+str(i)+']': vec}, step=t*config.n_workers)
                wandb.log({'weighted_rew_mean ['+str(i)+']': w_posterior_mean[i] * vec}, step=t*config.n_workers)

            # Log Objectives
            obj_ret = np.array(objective_logs_sum)
            obj_ret_logs = np.mean(obj_ret, axis=0)
            for i, ret in enumerate(obj_ret_logs):
                wandb.log({'Obj_' + str(i): ret}, step=t*config.n_workers)

            # Log total weighted sum
            wandb.log({'Returns mean': np.mean(dataset.log_rewards())}, step=t*config.n_workers)

            # Update Models
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs, config.entropy_reg)

            # rew_a, rew_b, logs_a, logs_b = volume_buffer.sample_return_pair_v2()
            if c["query_selection"] == "random":
                observed_rew_a, observed_rew_b, ret_a, ret_b = volume_buffer.sample_return_pair_no_batch_reset()
            elif c["query_selection"] == "random_no_double_null":
                observed_rew_a, observed_rew_b, ret_a, ret_b = volume_buffer.sample_return_pair_no_batch_reset_no_double_zeros()
            elif c["query_selection"] == "random_less_null":
                observed_rew_a, observed_rew_b, ret_a, ret_b = volume_buffer.sample_return_pair_no_batch_reset_less_zeros_no_double()
            elif c["query_selection"] == "compare_EUS":
                for k in range(c["nb_query_test"]):
                    volume_buffer.compare_EUS(w_posterior, w_posterior_mean, c["prop_w_mode"], c["posterior_mode"], preference_learner)
                ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()
            elif c["query_selection"] == "compare_EUS_less_zeros":
                for k in range(c["nb_query_test"]):
                    volume_buffer.compare_EUS(w_posterior, w_posterior_mean, c["prop_w_mode"], c["posterior_mode"], preference_learner, sample_mode="less_zeros")
                ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()
            elif c["query_selection"] == "compare_MORAL":
                for k in range(c["nb_query_test"]):
                    volume_buffer.compare_MORAL(w_posterior)
                ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()
            elif c["query_selection"] == "compare_MORAL_less_zeros":
                for k in range(c["nb_query_test"]):
                    volume_buffer.compare_MORAL(w_posterior, sample_mode="less_zeros")
                ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()
            elif c["query_selection"] == "compare_basic_log_lik":
                for k in range(c["nb_query_test"]):
                    volume_buffer.compare_basic_log_lik(w_posterior, config.temperature_mcmc)
                ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()
            elif c["query_selection"] == "compare_basic_log_lik_less_zeros":
                for k in range(c["nb_query_test"]):
                    volume_buffer.compare_basic_log_lik(w_posterior, config.temperature_mcmc, sample_mode="less_zeros")
            ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()

            volume_buffer.best_returns = (ret_a, ret_b)
            volume_buffer.best_observed_returns = (observed_rew_a, observed_rew_b)
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