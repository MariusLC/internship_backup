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


def query_pair(ret_a, ret_b, dimension_pref, RATIO_NORMALIZED, RATIO_linalg_NORMALIZED, norm="sum"):
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
	# print("ret_a_normalized = ", ret_a_normalized)
	# print("ret_b_normalized = ", ret_b_normalized)
	# print("ret_a_linalg_normalized = ", ret_a_linalg_normalized)
	# print("ret_b_linalg_normalized = ", ret_b_linalg_normalized)
	# print("self.ratio_normalized = ", RATIO_NORMALIZED)

	kl_a_sum = st.entropy(ret_a_normalized, RATIO_NORMALIZED)
	kl_b_sum = st.entropy(ret_b_normalized, RATIO_NORMALIZED)
	kl_a_linalg = st.entropy(ret_a_linalg_normalized, RATIO_linalg_NORMALIZED)
	kl_b_linalg = st.entropy(ret_b_linalg_normalized, RATIO_linalg_NORMALIZED)
	# print("kl_a = ", kl_a)
	# print("kl_b = ", kl_b)
	# print("kl_a_linalg = ", kl_a_linalg)
	# print("kl_b_linalg = ", kl_b_linalg)

	if norm == "sum":
		kl_a = kl_a_sum
		kl_b = kl_b_sum
	elif norm == "linalg":
		kl_a = kl_a_linalg
		kl_b = kl_b_linalg

	if kl_a < kl_b:
		preference = 1
	elif kl_b < kl_a:
		preference = -1
	else:
		preference = 1 if np.random.rand() < 0.5 else -1
	return preference, kl_a, kl_b

def estimate_vectorized_rew(env, agent, dataset, discriminator_list, gamma, eth_norm, non_eth_norm, env_steps=1000):
	states = env.reset()
	states_tensor = torch.tensor(states).float().to(device)
	for t in tqdm(range(env_steps)):
		actions, log_probs = agent.act(states_tensor)
		next_states, rewards, done, info = env.step(actions)

		airl_state = torch.tensor(states).to(device).float()
		airl_next_state = torch.tensor(next_states).to(device).float()

		airl_rewards_list = []
		for j in range(len(discriminator_list)):
			airl_rewards_list.append(discriminator_list[j].forward(airl_state, airl_next_state, gamma, eth_norm).squeeze(1))

		for j in range(len(discriminator_list)):
			airl_rewards_list[j] = airl_rewards_list[j].detach().cpu().numpy() * [0 if i else 1 for i in done]

		airl_rewards_array = np.array(airl_rewards_list)
		new_airl_rewards = [airl_rewards_array[:,i] for i in range(len(airl_rewards_list[0]))]
		train_ready = dataset.write_tuple_norm(states, actions, None, rewards, new_airl_rewards, done, log_probs)

		states = next_states.copy()
		states_tensor = torch.tensor(states).float().to(device)

	mean_returns = np.array(dataset.log_returns_sum()).mean(axis=0)
	w = [0.5 for j in range(len(discriminator_list)+1)]
	mean_vectorized_rewards = dataset.compute_scalarized_rewards(w, non_eth_norm, None) # wandb
	# volume_buffer.log_rewards_sum(dataset.log_vectorized_rew_sum())

	dataset.reset_trajectories()

	return mean_returns, mean_vectorized_rewards


# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_TEST_PREF_2.yaml"

if __name__ == '__main__':

	c = load_config(CONFIG_PATH, CONFIG_FILENAME)

	wandb.init(project='Test_preferences_2',
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

	volume_buffer = VolumeBuffer(len(c["ratio"]))

	# Create Environment
	env = VecEnv(env_id, config.n_workers)
	# env = GymWrapper(env_id)
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
		args = discriminator_list[i].estimate_normalisation_points(c["normalization_eth_sett"], rand_agent, generator_list[i], env_id, c["gamma"], steps=1000)
		discriminator_list[i].set_eval()

	dataset = TrajectoryDataset(batch_size=c["batchsize_ppo"], n_workers=c["n_workers"])
	# test
	# dataset.estimate_normalisation_points(c["normalization_non_eth_sett"], non_eth_expert, env_id, steps=1000)
	dataset.estimate_normalisation_points(c["normalization_non_eth_sett"], non_eth_expert, env_id, steps=10000)
	
	obj_rew, vect_rew = estimate_vectorized_rew(env, agent_test, dataset, discriminator_list, config.gamma, config.normalization_eth_sett, config.normalization_non_eth_sett, env_steps=1000)
	obj_rew_norm_sum = obj_rew / sum(obj_rew)
	obj_rew_norm_linalg = obj_rew / np.linalg.norm(obj_rew)
	print("mean objective reward expert = ", obj_rew)
	print("mean airl vectorized reward expert = ", vect_rew)

	# test
	# preference_learner = PreferenceLearner(d=c["dimension_pref"], n_iter=1000, warmup=100, temperature=config.temperature_mcmc, cov_range=config.cov_range, prior=config.prior)
	preference_learner = PreferenceLearner(d=c["dimension_pref"], n_iter=10000, warmup=1000, temperature=config.temperature_mcmc, cov_range=config.cov_range, prior=config.prior)

	w_posterior = preference_learner.sample_w_prior(preference_learner.n_iter)
	w_posterior_mean_uniform = w_posterior.mean(axis=0)

	HOF = []

	RATIO_NORMALIZED = c["ratio"]/np.sum(c["ratio"])
	RATIO_linalg_NORMALIZED = c["ratio"]/np.linalg.norm(c["ratio"])

	# objective_returns = []
	# observed_rewards = []
	# nb_traj = 0
	print("\n")
	train_ready = False
	while not train_ready:

		# Environment interaction
		actions, log_probs = agent_test.act(states_tensor)
		next_states, rewards, done, info = env.step(actions)
		# print("done = ", done)

		# Fetch AIRL rewards
		airl_state = torch.tensor(states).to(device).float()
		airl_next_state = torch.tensor(next_states).to(device).float()


		airl_rewards_list = []
		for j in range(c["nb_experts"]):
			airl_rewards_list.append(discriminator_list[j].forward(airl_state, airl_next_state, c["gamma"], c["normalization_eth_sett"]).squeeze(1))

		for j in range(c["nb_experts"]):
			airl_rewards_list[j] = airl_rewards_list[j].detach().cpu().numpy() * [0 if i else 1 for i in done]
			# airl_rewards_list[j] = airl_rewards_list[j] * (not done)

		airl_rewards_array = np.array(airl_rewards_list)
		new_airl_rewards = [airl_rewards_array[:,i] for i in range(len(airl_rewards_list[0]))]
		train_ready = dataset.write_tuple_norm(states, actions, None, rewards, new_airl_rewards, done, log_probs)

		# Prepare state input for next time step
		states = next_states.copy()
		states_tensor = torch.tensor(states).float().to(device)

	print("nb_traj = ", len(dataset.trajectories))
	volume_buffer.log_statistics_sum(dataset.log_returns_sum())
	mean_vectorized_rewards = dataset.compute_scalarized_rewards(w_posterior_mean_uniform, c["normalization_non_eth_sett"], None)
	volume_buffer.log_rewards_sum(dataset.log_vectorized_rew_sum())

	for i in range(c["n_queries"]):

		if c["query_selection"] == "random":
			observed_rew_a, observed_rew_b, ret_a, ret_b = volume_buffer.sample_return_pair_no_batch_reset()
		elif c["query_selection"] == "compare_EUS":
			for k in range(c["nb_query_test"]):
				volume_buffer.compare_EUS(w_posterior, w_posterior_mean_uniform, preference_learner)
			ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()
		elif c["query_selection"] == "compare_MORAL":
			for k in range(c["nb_query_test"]):
				volume_buffer.compare_MORAL(w_posterior)
			ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()
		elif c["query_selection"] == "compare_basic_log_lik":
			for k in range(c["nb_query_test"]):
				volume_buffer.compare_delta_basic_log_lik(w_posterior, config.temperature_mcmc)
			ret_a, ret_b, observed_rew_a, observed_rew_b = volume_buffer.get_best()

		# ret_a = objective_returns[0]
		# ret_b = objective_returns[1]
		# observed_rew_a = np.array(observed_rewards[0])
		# observed_rew_b = np.array(observed_rewards[1])
		delta = observed_rew_a - observed_rew_b

		# observed_rew_a_norm = observed_rew_a/np.linalg.norm(observed_rew_a)
		# observed_rew_b_norm = observed_rew_b/np.linalg.norm(observed_rew_b)
		print("ret_a = ",ret_a)
		print("ret_b = ",ret_b)
		print("observed_rew_a = ",observed_rew_a)
		print("observed_rew_b = ",observed_rew_b)
		# print("observed_rew_a_norm = ",observed_rew_a_norm)
		# print("observed_rew_b_norm = ",observed_rew_b_norm)
		print("delta = ",delta)

		# go query the preference expert
		preference, kl_a, kl_b = query_pair(ret_a, ret_b, c["dimension_pref"], RATIO_NORMALIZED, RATIO_linalg_NORMALIZED, norm="sum")
		print(preference)

		HOF.append((kl_a, ret_a, observed_rew_a))
		HOF.append((kl_b, ret_b, observed_rew_b))

		# Run MCMC on expert preferences
		preference_learner.log_preference(delta, preference)
		# preference_learner.log_returns(ret_a[:-1], ret_b[:-1])
		preference_learner.log_returns(observed_rew_a, observed_rew_b)


		w_posterior = []
		w_posterior_mean_temp = w_posterior_mean_uniform
		# nb_mcmc = 10
		nb_mcmc = 1
		if config.mcmc_log == "active":
			if config.mcmc_type == "parallel":
				for j in range(nb_mcmc):
					w_posterior_temp = preference_learner.mcmc_test(w_posterior_mean_uniform, c["prop_w_mode"], c["posterior_mode"], step=i*nb_mcmc+j)
					if j == 0 : 
						w_posterior = w_posterior_temp
					else :
						w_posterior = np.concatenate((w_posterior, w_posterior_temp))
					w_posterior_mean_temp = w_posterior_temp.mean(axis=0)
					# w_posterior_mean_temp = w_posterior_mean_temp/(np.linalg.norm(w_posterior_mean_temp) + 1e-15)

					print("NORM = ", np.linalg.norm(w_posterior_mean_temp))
					if np.linalg.norm(w_posterior_mean_temp) > 1:
						print(w_posterior_mean_temp)
					if (w_posterior_mean_temp <0).any():
						print("\n negative objective")
						print(w_posterior_mean_temp)

			elif config.mcmc_type == "successive":
				for j in range(nb_mcmc):
					w_posterior_temp = preference_learner.mcmc_test(w_posterior_mean_temp, c["prop_w_mode"], c["posterior_mode"], step=i*nb_mcmc+j)
					w_posterior = w_posterior_temp
					w_posterior_mean_temp = w_posterior_temp.mean(axis=0)
					# w_posterior_mean_temp = w_posterior_mean_temp/(np.linalg.norm(w_posterior_mean_temp) + 1e-15)

					print("NORM = ", np.linalg.norm(w_posterior_mean_temp))
					if np.linalg.norm(w_posterior_mean_temp) > 1:
						print(w_posterior_mean_temp)
					if (w_posterior_mean_temp <0).any():
						print("\n negative objective")
						print(w_posterior_mean_temp)

			elif config.mcmc_type == "concat":
				for j in range(nb_mcmc):
					w_posterior_temp = preference_learner.mcmc_test(w_posterior_mean_temp, c["prop_w_mode"], c["posterior_mode"], step=i*nb_mcmc+j)
					if j == 0 : 
						w_posterior = w_posterior_temp
					else :
						w_posterior = np.concatenate((w_posterior, w_posterior_temp))
					w_posterior_mean_temp = w_posterior_temp.mean(axis=0)
					# w_posterior_mean_temp = w_posterior_mean_temp/(np.linalg.norm(w_posterior_mean_temp) + 1e-15)

					print("NORM = ", np.linalg.norm(w_posterior_mean_temp))
					if np.linalg.norm(w_posterior_mean_temp) > 1:
						print(w_posterior_mean_temp)
					if (w_posterior_mean_temp <0).any():
						print("\n negative objective")
						print(w_posterior_mean_temp)

			w_posterior_mean = np.array(w_posterior).mean(axis=0)
			print("w_posterior_mean = ", w_posterior_mean)
			if sum(w_posterior_mean) != 0: 

				# making a 1 norm vector from w_posterior
				# w_posterior_mean = w_posterior_mean/np.linalg.norm(w_posterior_mean)
				w_posterior_mean = w_posterior_mean/(np.linalg.norm(w_posterior_mean) + 1e-15)

				# # normalize the vector 
				# w_posterior_mean = w_posterior_mean/np.sum(w_posterior_mean)
				
				print(f'New Posterior Mean {w_posterior_mean}')
			else :
				print(f'Keep the current Posterior Mean {w_posterior_mean}')


			weighted_obj_rew = w_posterior_mean * obj_rew[:len(w_posterior_mean)]
			weighted_obj_rew_sum = w_posterior_mean * obj_rew_norm_sum[:len(w_posterior_mean)]
			weighted_obj_rew_linalg = w_posterior_mean * obj_rew_norm_linalg[:len(w_posterior_mean)]
			weighted_airl_rew = w_posterior_mean * vect_rew[:len(w_posterior_mean)]

			distance_obj_sum = sum([(weighted_obj_rew_sum[j] - RATIO_NORMALIZED[j])**2 for j in range(len(RATIO_NORMALIZED))])
			distance_obj_linalg = sum([(weighted_obj_rew_linalg[j] - RATIO_NORMALIZED[j])**2 for j in range(len(RATIO_NORMALIZED))])
			distance_airl = sum([(weighted_airl_rew[j] - RATIO_NORMALIZED[j])**2 for j in range(len(RATIO_NORMALIZED))])

			# kl_a_sum = st.entropy(ret_a_normalized, RATIO_NORMALIZED)
			# kl_b_linalg = st.entropy(ret_b_linalg_normalized, RATIO_linalg_NORMALIZED)


			for j in range(len(w_posterior_mean)):
				wandb.log({'w_posterior_mean['+str(j)+"]": w_posterior_mean[j]}, step=(i+1)*nb_mcmc)
				wandb.log({'weighted_airl_rew ['+str(j)+']': weighted_airl_rew[j]}, step=(i+1)*nb_mcmc)
			wandb.log({'distance_obj_sum_to_ratio': distance_obj_sum}, step=(i+1)*nb_mcmc)
			wandb.log({'distance_obj_linalg_to_ratio': distance_obj_linalg}, step=(i+1)*nb_mcmc)
			wandb.log({'distance_airl_to_ratio': distance_airl}, step=(i+1)*nb_mcmc)


		elif config.mcmc_log == "final" and i == c["n_queries"]-1:
			if config.mcmc_type == "parallel":
				for j in range(nb_mcmc):
					w_posterior_temp = preference_learner.mcmc_test(w_posterior_mean_uniform, c["prop_w_mode"], c["posterior_mode"], step=i*nb_mcmc+j)
					if j == 0 : 
						w_posterior = w_posterior_temp
					else :
						w_posterior = np.concatenate((w_posterior, w_posterior_temp))
					w_posterior_mean_temp = w_posterior_temp.mean(axis=0)
					# w_posterior_mean_temp = w_posterior_mean_temp/(np.linalg.norm(w_posterior_mean_temp) + 1e-15)

					print("NORM = ", np.linalg.norm(w_posterior_mean_temp))
					if np.linalg.norm(w_posterior_mean_temp) > 1:
						print(w_posterior_mean_temp)
					if (w_posterior_mean_temp <0).any():
						print("\n negative objective")
						print(w_posterior_mean_temp)

			elif config.mcmc_type == "successive":
				for j in range(nb_mcmc):
					w_posterior_temp = preference_learner.mcmc_test(w_posterior_mean_temp, c["prop_w_mode"], c["posterior_mode"], step=i*nb_mcmc+j)
					w_posterior = w_posterior_temp
					w_posterior_mean_temp = w_posterior_temp.mean(axis=0)
					# w_posterior_mean_temp = w_posterior_mean_temp/(np.linalg.norm(w_posterior_mean_temp) + 1e-15)

					print("NORM = ", np.linalg.norm(w_posterior_mean_temp))
					if np.linalg.norm(w_posterior_mean_temp) > 1:
						print(w_posterior_mean_temp)
					if (w_posterior_mean_temp <0).any():
						print("\n negative objective")
						print(w_posterior_mean_temp)

			elif config.mcmc_type == "concat":
				for j in range(nb_mcmc):
					w_posterior_temp = preference_learner.mcmc_test(w_posterior_mean_temp, c["prop_w_mode"], c["posterior_mode"], step=i*nb_mcmc+j)
					if j == 0 : 
						w_posterior = w_posterior_temp
					else :
						w_posterior = np.concatenate((w_posterior, w_posterior_temp))
					w_posterior_mean_temp = w_posterior_temp.mean(axis=0)
					# w_posterior_mean_temp = w_posterior_mean_temp/(np.linalg.norm(w_posterior_mean_temp) + 1e-15)

					print("NORM = ", np.linalg.norm(w_posterior_mean_temp))
					if np.linalg.norm(w_posterior_mean_temp) > 1:
						print(w_posterior_mean_temp)
					if (w_posterior_mean_temp <0).any():
						print("\n negative objective")
						print(w_posterior_mean_temp)

			w_posterior_mean = np.array(w_posterior).mean(axis=0)
			print("w_posterior_mean = ", w_posterior_mean)
			if sum(w_posterior_mean) != 0: 

				# making a 1 norm vector from w_posterior
				# w_posterior_mean = w_posterior_mean/np.linalg.norm(w_posterior_mean)
				w_posterior_mean = w_posterior_mean/(np.linalg.norm(w_posterior_mean) + 1e-15)

				# # normalize the vector 
				# w_posterior_mean = w_posterior_mean/np.sum(w_posterior_mean)
				
				print(f'New Posterior Mean {w_posterior_mean}')
			else :
				print(f'Keep the current Posterior Mean {w_posterior_mean}')


			weighted_obj_rew = w_posterior_mean * obj_rew[:len(w_posterior_mean)]
			weighted_obj_rew_sum = w_posterior_mean * obj_rew_norm_sum[:len(w_posterior_mean)]
			weighted_obj_rew_linalg = w_posterior_mean * obj_rew_norm_linalg[:len(w_posterior_mean)]
			weighted_airl_rew = w_posterior_mean * vect_rew[:len(w_posterior_mean)]

			distance_obj_sum = sum([(weighted_obj_rew_sum[j] - RATIO_NORMALIZED[j])**2 for j in range(len(RATIO_NORMALIZED))])
			distance_obj_linalg = sum([(weighted_obj_rew_linalg[j] - RATIO_NORMALIZED[j])**2 for j in range(len(RATIO_NORMALIZED))])
			distance_airl = sum([(weighted_airl_rew[j] - RATIO_NORMALIZED[j])**2 for j in range(len(RATIO_NORMALIZED))])

			# kl_a_sum = st.entropy(ret_a_normalized, RATIO_NORMALIZED)
			# kl_b_linalg = st.entropy(ret_b_linalg_normalized, RATIO_linalg_NORMALIZED)


			for j in range(len(w_posterior_mean)):
				wandb.log({'w_posterior_mean['+str(j)+"]": w_posterior_mean[j]}, step=(i+1)*nb_mcmc)
				wandb.log({'weighted_airl_rew ['+str(j)+']': weighted_airl_rew[j]}, step=(i+1)*nb_mcmc)
			wandb.log({'distance_obj_sum_to_ratio': distance_obj_sum}, step=(i+1)*nb_mcmc)
			wandb.log({'distance_obj_linalg_to_ratio': distance_obj_linalg}, step=(i+1)*nb_mcmc)
			wandb.log({'distance_airl_to_ratio': distance_airl}, step=(i+1)*nb_mcmc)


		# # Reset PPO buffer
		# dataset.reset_trajectories()
		# volume_buffer.reset()
		# volume_buffer.reset_batch() # Do we need to reset batch every time ?

	dtype = [("kl_a", float), ("ret_a", np.float64, (4,)), ("observed_rew_a", np.float64, (3,))]
	HOF = np.array(HOF, dtype=dtype)[:20]
	HOF_sorted = np.sort(HOF, order="kl_a")
	print("HOF = ", HOF_sorted)


	