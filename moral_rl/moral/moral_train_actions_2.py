import numpy as np
import torch
import math
import scipy.stats as st
from torch.distributions import *
from envs.gym_wrapper import *
from moral.ppo import *
from utils.load_config import *
from moral.airl import *
from moral.active_learning import *
from tqdm import tqdm
import pickle
from moral.preference_giver import *

def estimate_vectorized_rew_2(traj_test):
	mean_returns = np.zeros(len(traj_test[0]["returns"][0]))
	mean_vectorized_rewards = np.zeros(len(traj_test[0]["vectorized_rewards"][0]))
	for traj in traj_test:
		mean_returns += np.sum(traj["returns"], axis=0)
		mean_vectorized_rewards += np.sum(traj["vectorized_rewards"], axis=0)
	mean_returns /= len(traj_test)
	mean_vectorized_rewards /= len(traj_test)
	return mean_returns, mean_vectorized_rewards

def run_mcmc(config, preference_learner, w_posterior_mean_uniform, RATIO_NORMALIZED, traj_test, preference_giver):
	w_posterior_mean_temp = w_posterior_mean_uniform
	if config.mcmc_type == "parallel":
		for j in range(config.nb_mcmc):
			w_posterior_temp = preference_learner.mcmc_test(w_posterior_mean_uniform, c["prop_w_mode"], c["posterior_mode"], step=i*config.nb_mcmc+j)
			if j == 0 : 
				w_posterior = w_posterior_temp
			else :
				w_posterior = np.concatenate((w_posterior, w_posterior_temp))
	elif config.mcmc_type == "successive":
		for j in range(config.nb_mcmc):
			w_posterior_temp = preference_learner.mcmc_test(w_posterior_mean_temp, c["prop_w_mode"], c["posterior_mode"], step=i*config.nb_mcmc+j)
			w_posterior = w_posterior_temp
			w_posterior_mean_temp = w_posterior_temp.mean(axis=0)
			# w_posterior_mean_temp = w_posterior_mean_temp/(np.linalg.norm(w_posterior_mean_temp) + 1e-15)
	elif config.mcmc_type == "concat":
		for j in range(config.nb_mcmc):
			w_posterior_temp = preference_learner.mcmc_test(w_posterior_mean_temp, c["prop_w_mode"], c["posterior_mode"], step=i*config.nb_mcmc+j)
			if j == 0 : 
				w_posterior = w_posterior_temp
			else :
				w_posterior = np.concatenate((w_posterior, w_posterior_temp))
			w_posterior_mean_temp = w_posterior_temp.mean(axis=0)
			# w_posterior_mean_temp = w_posterior_mean_temp/(np.linalg.norm(w_posterior_mean_temp) + 1e-15)
	
	# Prints and Logs
	w_posterior_mean = np.array(w_posterior).mean(axis=0)
	print("w_posterior_mean before norm = ", w_posterior_mean)
	if sum(w_posterior_mean) != 0: 
		w_posterior_mean = w_posterior_mean/(np.linalg.norm(w_posterior_mean) + 1e-15)
		print(f'New Posterior Mean {w_posterior_mean}')
	else :
		print(f'Keep the current Posterior Mean {w_posterior_mean}')

	obj_rew, vect_rew = estimate_vectorized_rew_2(traj_test)
	obj_rew_norm_sum = obj_rew / sum(obj_rew)
	obj_rew_norm_linalg = obj_rew / np.linalg.norm(obj_rew)
	print("mean objective reward expert = ", obj_rew)
	print("mean airl vectorized reward expert = ", vect_rew)

	weighted_obj_rew = w_posterior_mean * obj_rew[:len(w_posterior_mean)]
	weighted_obj_rew_sum = w_posterior_mean * obj_rew_norm_sum[:len(w_posterior_mean)]
	weighted_obj_rew_linalg = w_posterior_mean * obj_rew_norm_linalg[:len(w_posterior_mean)]
	weighted_airl_rew = w_posterior_mean * vect_rew[:len(w_posterior_mean)]

	distance_obj_sum = sum([(weighted_obj_rew_sum[j] - RATIO_NORMALIZED[j])**2 for j in range(len(RATIO_NORMALIZED))])
	distance_obj_linalg = sum([(weighted_obj_rew_linalg[j] - RATIO_NORMALIZED[j])**2 for j in range(len(RATIO_NORMALIZED))])
	distance_airl = sum([(weighted_airl_rew[j] - RATIO_NORMALIZED[j])**2 for j in range(len(RATIO_NORMALIZED))])

	for j in range(len(w_posterior_mean)):
		wandb.log({'w_posterior_mean['+str(j)+"]": w_posterior_mean[j]}, step=0)
		wandb.log({'weighted_airl_rew ['+str(j)+']': weighted_airl_rew[j]}, step=0)
	wandb.log({'distance_obj_sum_to_ratio': distance_obj_sum}, step=0)
	wandb.log({'distance_obj_linalg_to_ratio': distance_obj_linalg}, step=0)
	wandb.log({'distance_airl_to_ratio': distance_airl}, step=0)

	# NEW WEIGHT QUALITY HEURISTIC
	weight_eval = preference_giver.evaluate_weights(config.n_best, w_posterior_mean, traj_test)
	weight_eval_10, weight_eval_10_norm = preference_giver.evaluate_weights_print(10, w_posterior_mean, traj_test)
	wandb.log({'weight_eval': weight_eval}, step=0)
	wandb.log({'weight_eval TOP 10': weight_eval_10}, step=0)
	wandb.log({'weight_eval norm TOP 10': weight_eval_10_norm}, step=0)

	# SCORE VS RANDOM WEIGHTS TO EVALUATE WEIGHTS QUALITY
	weight_eval_rand = []
	for j in range(100):
		weights = st.multivariate_normal(mean=np.ones(3)/np.linalg.norm(np.ones(3)), cov=0.01).rvs()
		weight_eval_rand.append(preference_giver.evaluate_weights(config.n_best, weights, traj_test))
	mean_weight_eval_rand = np.mean(weight_eval_rand)
	median_weight_eval_rand = np.median(weight_eval_rand)
	min_weight_eval_rand = min(weight_eval_rand)
	max_weight_eval_rand = max(weight_eval_rand)
	norm_score_vs_rand = (weight_eval - min_weight_eval_rand) / (max_weight_eval_rand - min_weight_eval_rand)
	print("mean_weight_eval_rand = ", mean_weight_eval_rand)
	print("min_weight_eval_rand = ", min_weight_eval_rand)
	print("max_weight_eval_rand = ", max_weight_eval_rand)
	print("median_weight_eval_rand = ", median_weight_eval_rand)
	print("norm_score_vs_rand = ", norm_score_vs_rand)
	wandb.log({'mean_weight_eval_rand': mean_weight_eval_rand}, step=0)
	wandb.log({'min_weight_eval_rand': min_weight_eval_rand}, step=0)
	wandb.log({'max_weight_eval_rand': max_weight_eval_rand}, step=0)
	wandb.log({'median_weight_eval_rand': median_weight_eval_rand}, step=0)
	wandb.log({'norm_score_vs_rand': norm_score_vs_rand}, step=0)

	return w_posterior_mean


# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_MORAL_2.yaml"

if __name__ == '__main__':

	c = load_config(CONFIG_PATH, CONFIG_FILENAME)

	wandb.init(project='MORAL',
		config=c)
	config=wandb.config

	generators_filenames = []
	discriminators_filenames = []
	moral_filename = c["data_path"]+c["env"]+"/"+c["vanilla_path"]+c["moral_path"]+str(c["experts_weights"])+c["special_name_agent"]+c["model_ext"]
	non_eth_expert_filename = c["data_path"]+c["env"]+"/"+str(c["non_eth_experts_weights"])+"/"+c["expe_path"]+c["model_ext"]
	for i in range(c["nb_experts"]):
		path = c["data_path"]+c["env"]+"/"+str(c["experts_weights"][i])+"/"
		generators_filenames.append(path+c["expe_path"]+c["model_ext"])
		discriminators_filenames.append(path+c["disc_path"]+c["model_ext"])

	env_steps = c["env_steps"]
	if c["real_params"]:
		env_steps = int(c["env_steps"]/c["n_workers"])

	volume_buffer = VolumeBuffer(len(c["ratio"]))

	# Create Environment
	env = VecEnv(config.env_id, config.n_workers)
	states = env.reset()
	states_tensor = torch.tensor(states).float().to(device)

	# Fetch Shapes
	n_actions = env.action_space.n
	obs_shape = env.observation_space.shape
	state_shape = obs_shape[:-1]
	in_channels = obs_shape[-1]

	# Initialize Models
	print('Initializing and Normalizing Rewards...')
	ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
	optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)

	# Collect test trajectories to evaluate mcmc weights quality
	traj_test = pickle.load(open(config.demos_filename, 'rb'))

	# Expert i
	discriminator_list = []
	generator_list = []
	# utop_list = []

	rand_agent = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
	non_eth_expert = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
	non_eth_expert.load_state_dict(torch.load(non_eth_expert_filename, map_location=torch.device('cpu')))

	for i in range(config.nb_experts):
		discriminator_list.append(Discriminator(state_shape=state_shape, in_channels=in_channels).to(device))
		discriminator_list[i].load_state_dict(torch.load(discriminators_filenames[i], map_location=torch.device('cpu')))
		generator_list.append(PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device))
		generator_list[i].load_state_dict(torch.load(generators_filenames[i], map_location=torch.device('cpu')))
		if config.test:
			args = discriminator_list[i].estimate_normalisation_points(config.eth_norm, rand_agent, generator_list[i], config.env_id, config.gamma, steps=1000) # tests
		else:
			args = discriminator_list[i].estimate_normalisation_points(config.eth_norm, rand_agent, generator_list[i], config.env_id, config.gamma, steps=10000)
		
		discriminator_list[i].set_eval()


	dataset = TrajectoryDataset(batch_size=config.batchsize_ppo_phase_1, n_workers=config.n_workers)
	if config.test:
		dataset.estimate_normalisation_points(config.non_eth_norm, non_eth_expert, config.env_id, steps=1000) # tests
	else :
		dataset.estimate_normalisation_points(config.non_eth_norm, non_eth_expert, config.env_id, steps=10000)

	# Active Learning
	if config.test:
		preference_learner = PreferenceLearner(d=len(config.experts_weights)+1, n_iter=1000, warmup=100, temperature=config.temperature_mcmc, cov_range=config.cov_range, prior=config.prior) # tests
	else :
		preference_learner = PreferenceLearner(d=len(config.experts_weights)+1, n_iter=10000, warmup=1000, temperature=config.temperature_mcmc, cov_range=config.cov_range, prior=config.prior)

	volume_buffer = VolumeBuffer(len(config.ratio))
	if config.pref_giver_no_null:
		preference_giver = PreferenceGiverv3_no_null(config.ratio)
	else :
		preference_giver = PreferenceGiverv3(config.ratio)


	w_posterior = preference_learner.sample_w_prior(preference_learner.n_iter)
	w_posterior_mean = w_posterior.mean(axis=0)
	# Log weight vector
	# for i in range(len(w_posterior_mean)):
		# wandb.log({'w_posterior_mean ['+str(i)+']': w_posterior_mean[i]}, step=0)

	RATIO_NORMALIZED = c["ratio"]/np.sum(c["ratio"])
	RATIO_linalg_NORMALIZED = c["ratio"]/np.linalg.norm(c["ratio"])




	##########################################################################################
	# PHASE 1 : Collect all preferences to estimate the expert's weights
	##########################################################################################

	train_ready = False
	while not train_ready:
		# Environment interaction
		actions, log_probs = ppo.act(states_tensor)
		next_states, rewards, done, info = env.step(actions)

		# Fetch AIRL rewards
		airl_state = torch.tensor(states).to(device).float()
		airl_next_state = torch.tensor(next_states).to(device).float()

		airl_rewards_list = []
		for j in range(c["nb_experts"]):
			airl_rewards_list.append(discriminator_list[j].forward(airl_state, airl_next_state, c["gamma"], c["eth_norm"]).squeeze(1))

		for j in range(c["nb_experts"]):
			airl_rewards_list[j] = airl_rewards_list[j].detach().cpu().numpy() * [0 if i else 1 for i in done]
			# airl_rewards_list[j] = airl_rewards_list[j] * (not done)

		airl_rewards_array = np.array(airl_rewards_list)
		new_airl_rewards = [airl_rewards_array[:,i] for i in range(len(airl_rewards_list[0]))]
		train_ready = dataset.write_tuple_norm(states, actions, None, rewards, new_airl_rewards, done, log_probs)

		# Prepare state input for next time step
		states = next_states.copy()
		states_tensor = torch.tensor(states).float().to(device)

	# log objective rewards into volume_buffer before normalizing it
	# volume_buffer.log_statistics_sum(dataset.log_returns_actions())
	volume_buffer.log_statistics_sum_print(dataset.log_returns_actions())
	mean_vectorized_rewards = dataset.compute_scalarized_rewards(w_posterior_mean, c["non_eth_norm"], None)
	volume_buffer.log_rewards_sum(dataset.log_vectorized_rew_actions())

	for i in range(c["n_queries"]):
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

		delta = observed_rew_a - observed_rew_b

		print("ret_a = ",ret_a)
		print("ret_b = ",ret_b)
		print("observed_rew_a = ",observed_rew_a)
		print("observed_rew_b = ",observed_rew_b)
		print("delta = ",delta)

		# go query the preference expert
		preference = preference_giver.query_pair(ret_a, ret_b)
		# print(preference)

		# save preferences in the preference learner
		preference_learner.log_preference(delta, preference)
		preference_learner.log_returns(observed_rew_a, observed_rew_b)

	# Calculate new w_posterior with all preferences
	w_posterior_mean = run_mcmc(config, preference_learner, w_posterior_mean, RATIO_NORMALIZED, traj_test, preference_giver)

	# Reset PPO buffer
	dataset.reset_trajectories()
	volume_buffer.reset()
	volume_buffer.reset_batch() 




	##########################################################################################
	# PHASE 2 : Run the MORL optimization with the fixed weights
	##########################################################################################

	# set the batch_size for phase 2
	dataset.batch_size = config.batchsize_ppo_phase_2

	for t in tqdm(range(env_steps)):

		# Environment interaction
		actions, log_probs = ppo.act(states_tensor)
		next_states, rewards, done, info = env.step(actions)

		# Fetch AIRL rewards
		airl_state = torch.tensor(states).to(device).float()
		airl_next_state = torch.tensor(next_states).to(device).float()

		airl_rewards_list = []
		for j in range(config.nb_experts):
			airl_rewards_list.append(discriminator_list[j].forward(airl_state, airl_next_state, config.gamma, config.eth_norm).squeeze(1).detach().cpu().numpy() * [0 if i else 1 for i in done])

		airl_rewards_array = np.array(airl_rewards_list)
		new_airl_rewards = [airl_rewards_array[:,i] for i in range(len(airl_rewards_list[0]))]
		train_ready = dataset.write_tuple_norm(states, actions, None, rewards, new_airl_rewards, done, log_probs)

		if train_ready:

			objective_logs_sum = dataset.log_returns_sum()
			mean_vectorized_rewards = dataset.compute_scalarized_rewards(w_posterior_mean, config.non_eth_norm, wandb)

			# Log mean vectorized rewards
			for i, vec in enumerate(mean_vectorized_rewards):
				wandb.log({'vectorized_rew_mean ['+str(i)+']': vec}, step=t*config.n_workers)
				wandb.log({'w_posterior_mean ['+str(i)+']': w_posterior_mean[i]}, step=t*config.n_workers)
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

			# Reset PPO buffer
			dataset.reset_trajectories()

		# Prepare state input for next time step
		states = next_states.copy()
		states_tensor = torch.tensor(states).float().to(device)

	save_data(ppo, moral_filename)
