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


def query_pair(ret_a, ret_b, dimension_pref, RATIO_NORMALIZED, RATIO_linalg_NORMALIZED):
	print("query_pair = "+str(ret_a)+" , "+str(ret_b))

	ret_a_copy = ret_a.copy()
	ret_b_copy = ret_b.copy()

	ret_a_normalized = []
	ret_b_normalized = []

	ret_a_linalg_normalized = []
	ret_b_linalg_normalized = []

	for i in range(dimension_pref):
		# To avoid numerical instabilities in KL
		ret_a_copy[i] += 1e-5
		ret_b_copy[i] += 1e-5

	ret_a_sum = sum(ret_a_copy)
	ret_b_sum = sum(ret_b_copy)

	ret_a_norm = np.linalg.norm(ret_a_copy)
	ret_b_norm = np.linalg.norm(ret_b_copy)

	for i in range(dimension_pref):
		ret_a_normalized.append(ret_a_copy[i]/ret_a_sum)
		ret_b_normalized.append(ret_b_copy[i]/ret_b_sum)

		ret_a_linalg_normalized.append(ret_a_copy[i]/ret_a_norm)
		ret_b_linalg_normalized.append(ret_b_copy[i]/ret_b_norm)

	# scipy.stats.entropy(pk, qk=None, base=None, axis=0) = S = sum(pk * log(pk / qk), axis=axis)
	print("ret_a_normalized = ", ret_a_normalized)
	print("ret_b_normalized = ", ret_b_normalized)
	print("ret_a_linalg_normalized = ", ret_a_linalg_normalized)
	print("ret_b_linalg_normalized = ", ret_b_linalg_normalized)
	print("self.ratio_normalized = ", RATIO_NORMALIZED)

	kl_a = st.entropy(ret_a_normalized, RATIO_NORMALIZED)
	kl_b = st.entropy(ret_b_normalized, RATIO_NORMALIZED)
	kl_a_linalg = st.entropy(ret_a_linalg_normalized, RATIO_linalg_NORMALIZED)
	kl_b_linalg = st.entropy(ret_b_linalg_normalized, RATIO_linalg_NORMALIZED)
	print("kl_a = ", kl_a)
	print("kl_b = ", kl_b)
	print("kl_a_linalg = ", kl_a_linalg)
	print("kl_b_linalg = ", kl_b_linalg)

	if kl_a < kl_b:
		preference = 1
	elif kl_b < kl_a:
		preference = -1
	else:
		preference = 1 if np.random.rand() < 0.5 else -1
	return preference, kl_a, kl_b



# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_TEST_PREF.yaml"

if __name__ == '__main__':

	c = load_config(CONFIG_PATH, CONFIG_FILENAME)

	wandb.init(project='Test_preferences',
		config=c)
	config=wandb.config

	generators_filenames = []
	discriminators_filenames = []
	non_eth_expert_filename = c["data_path"]+c["env"]+"/"+str(c["non_eth_experts_weights"])+"/"+c["expe_path"]+c["model_ext"]
	for i in range(c["nb_experts"]):
		path = c["data_path"]+c["env"]+"/"+str(c["experts_weights"][i])+"/"
		generators_filenames.append(path+c["expe_path"]+c["model_ext"])
		discriminators_filenames.append(path+c["disc_path"]+c["model_ext"])

	env_id = c["env_rad"]+c["env"]

	# Create Environment
	env = GymWrapper(env_id)
	states = env.reset()
	states_tensor = torch.tensor(states).float().to(device)

	# Fetch Shapes
	n_actions = env.action_space.n
	obs_shape = env.observation_space.shape
	state_shape = obs_shape[:-1]
	in_channels = obs_shape[-1]


	# get an agent to act on the environment
	# agent_test_name = "generated_data/v3/moral_agents/[[0, 1, 0, 1], [0, 0, 1, 1]]131_new_norm_v6_v3_after_queries_fixed.pt"
	agent_test = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions)
	agent_test.load_state_dict(torch.load(c["agent_test_name"], map_location=torch.device('cpu')))

	#Expert i
	discriminator_list = []
	generator_list = []
	rand_agent = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
	non_eth_expert = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
	non_eth_expert.load_state_dict(torch.load(non_eth_expert_filename, map_location=torch.device('cpu')))
	for i in range(c["nb_experts"]):
		discriminator_list.append(Discriminator(state_shape=state_shape, in_channels=in_channels).to(device))
		discriminator_list[i].load_state_dict(torch.load(discriminators_filenames[i], map_location=torch.device('cpu')))
		generator_list.append(PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device))
		generator_list[i].load_state_dict(torch.load(generators_filenames[i], map_location=torch.device('cpu')))
		args = discriminator_list[i].estimate_normalisation_points(c["normalization_eth_sett"], rand_agent, generator_list[i], env_id, c["gamma"], steps=10000)
		discriminator_list[i].set_eval()

	dataset = TrajectoryDataset(batch_size=c["batchsize_ppo"], n_workers=c["n_workers"])
	dataset.estimate_normalisation_points(c["normalization_non_eth_sett"], non_eth_expert, env_id, steps=10000)



	preference_learner = PreferenceLearner(d=c["dimension_pref"], n_iter=10000, warmup=1000, temperature=config.temperature_mcmc)

	w_posterior = preference_learner.sample_w_prior(preference_learner.n_iter)
	w_posterior_mean = w_posterior.mean(axis=0)

	HOF = []

	RATIO_NORMALIZED = c["ratio"]/np.sum(c["ratio"])
	RATIO_linalg_NORMALIZED = c["ratio"]/np.linalg.norm(c["ratio"])

	for i in range(c["n_queries"]): 

		# objective_returns = []
		# observed_rewards = []
		# nb_traj = 0
		print("\n")
		train_ready = False
		while not train_ready:
			curr_objective_returns = []
			curr_observed_rewards = []

			# Environment interaction
			actions, log_probs = agent_test.act(states_tensor)
			next_states, rewards, done, info = env.step(actions)

			# Fetch AIRL rewards
			airl_state = torch.tensor(states).to(device).float()
			airl_next_state = torch.tensor(next_states).to(device).float()


			airl_rewards_list = []
			for j in range(c["nb_experts"]):
				airl_rewards_list.append(discriminator_list[j].forward(airl_state, airl_next_state, c["gamma"], c["normalization_eth_sett"]).squeeze(1).item())

			for j in range(c["nb_experts"]):
				airl_rewards_list[j] = airl_rewards_list[j] * (not done)

			# airl_rewards_array = np.array([rewards[0]]+airl_rewards_list)
			airl_rewards_array = airl_rewards_array = np.array(airl_rewards_list)

			train_ready = dataset.write_tuple_norm([states], [actions], None, [rewards], [airl_rewards_array], [done], [log_probs])

			# curr_objective_returns.append(rewards)
			# curr_observed_rewards.append(airl_rewards_array)

			if done :
				# array_rew = np.array(curr_observed_rewards)
				# objective_returns.append(curr_objective_returns)
				# observed_rewards.append(curr_observed_rewards)
				# nb_traj += 1
				env.reset()

			# Prepare state input for next time step
			states = next_states.copy()
			states_tensor = torch.tensor(states).float().to(device)

		objective_returns = dataset.log_returns_sum()
		dataset.compute_scalarized_rewards(w_posterior_mean, c["normalization_non_eth_sett"], None)
		observed_rewards = dataset.log_vectorized_rew_sum()


		# GET RET AND REW and delta rew
		# ret_a = [5, 5, 5]
		# ret_b = [5, 10, 5]
		ret_a = objective_returns[0]
		ret_b = objective_returns[1]
		observed_rew_a = np.array(observed_rewards[0])
		observed_rew_b = np.array(observed_rewards[1])
		delta = observed_rew_a - observed_rew_b

		observed_rew_a_norm = observed_rew_a/np.linalg.norm(observed_rew_a)
		observed_rew_b_norm = observed_rew_b/np.linalg.norm(observed_rew_b)
		print("ret_a = ",ret_a)
		print("ret_b = ",ret_b)
		print("observed_rew_a = ",observed_rew_a)
		print("observed_rew_b = ",observed_rew_b)
		print("observed_rew_a_norm = ",observed_rew_a_norm)
		print("observed_rew_b_norm = ",observed_rew_b_norm)
		print("delta = ",delta)

		# go query the preference expert
		preference, kl_a, kl_b = query_pair(ret_a[:-1], ret_b[:-1], c["dimension_pref"], RATIO_NORMALIZED, RATIO_linalg_NORMALIZED)
		print(preference)

		HOF.append((kl_a, ret_a, observed_rew_a))
		HOF.append((kl_b, ret_b, observed_rew_b))

		# Run MCMC on expert preferences
		preference_learner.log_preference(delta, preference)
		# preference_learner.log_returns(ret_a[:-1], ret_b[:-1])
		preference_learner.log_returns(observed_rew_a, observed_rew_b)
		print("w_posterior_mean = ", w_posterior_mean)
		w_posterior = preference_learner.mcmc_test(w_posterior_mean, c["posterior_mode"], c["prop_w_mode"])
		w_posterior_mean = w_posterior.mean(axis=0)
		print("w_posterior_mean = ", w_posterior_mean)
		if sum(w_posterior_mean) != 0: 

			# making a 1 norm vector from w_posterior
			w_posterior_mean = w_posterior_mean/np.linalg.norm(w_posterior_mean)

			# # normalize the vector 
			# w_posterior_mean = w_posterior_mean/np.sum(w_posterior_mean)
			
			print(f'New Posterior Mean {w_posterior_mean}')
		else :
			print(f'Keep the current Posterior Mean {w_posterior_mean}')

		for j in range(len(w_posterior_mean)):
			wandb.log({'w_posterior_mean['+str(j)+"]": w_posterior_mean[j]}, step=i)

		# Reset PPO buffer
		dataset.reset_trajectories()

	dtype = [("kl_a", float), ("ret_a", np.float64, (4,)), ("observed_rew_a", np.float64, (3,))]
	HOF = np.array(HOF, dtype=dtype)
	HOF_sorted = np.sort(HOF, order="kl_a")
	print("HOF = ", HOF_sorted)