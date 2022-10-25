# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
# from envs.gym_wrapper import *

# from tqdm import tqdm
import torch
# import matplotlib.pyplot as plt
import wandb
# import argparse
# import yaml
# import os

import math
# import scipy.stats as st

# from moral.active_learning import *
# from moral.preference_giver import *

# import math

from torch.distributions import *

from moral.preference_giver import *
import itertools
import random
from drlhp.preference_model import *
import time

import matplotlib.pyplot as plt


def check_null(ret_a, ret_b):
	if all(ret_b == 0) and any(ret_a > 0):
		return 1
	elif all(ret_a == 0) and any(ret_b > 0):
		return -1
	else:
		return 0.5

def query_pair_no_null(ret_a, ret_b, dimension_pref, RATIO_NORMALIZED):
	ret_a_copy = np.array(ret_a.copy())[:dimension_pref]+1e-5
	ret_b_copy = np.array(ret_b.copy())[:dimension_pref]+1e-5
	ret_a_normalized = ret_a_copy/sum(ret_a_copy)
	ret_b_normalized = ret_b_copy/sum(ret_b_copy)
	kl_a = st.entropy(ret_a_normalized, RATIO_NORMALIZED)
	kl_b = st.entropy(ret_b_normalized, RATIO_NORMALIZED)
	check = check_null(ret_a, ret_b)
	if check != 0.5:
		return check, kl_a, kl_b
	else :
		if kl_a < kl_b:
			preference = 1
		elif kl_b < kl_a:
			preference = -1
		else:
			preference = 1 if np.random.rand() < 0.5 else -1
		return preference, kl_a, kl_b


if __name__ == '__main__':


	# env_dim = 4
	# model_actions = "generated_data/v3/pref_model/ALLCOMBI_5b_2000e_[3, 1, 0, 2].pt"
	# preference_model_actions = PreferenceModelTEST(env_dim).to(device)
	# preference_model_actions.load_state_dict(torch.load(model_actions, map_location=torch.device('cpu')))
	# model_trajectories = "generated_data/v3/pref_model/trajectories/ALLCOMBI_100q_5b_2000e_[3, 1, 0, 2].pt"
	# preference_model_trajectories = PreferenceModelTEST(env_dim).to(device)
	# preference_model_trajectories.load_state_dict(torch.load(model_trajectories, map_location=torch.device('cpu')))

	# n_queries = 10
	# for i in range(n_queries):
	# 	ret_a = randomlist = random.sample(range(0, 13), 3) + [random.randint(-8, 0)]
	# 	ret_b = randomlist = random.sample(range(0, 13), 3) + [random.randint(-8, 0)]
	# 	evaluation_actions_a = preference_model_actions.evaluate_action_detach(ret_a)
	# 	evaluation_trajectories_a = preference_model_trajectories.evaluate_action_detach(ret_a)
	# 	evaluation_actions_b = preference_model_actions.evaluate_action_detach(ret_b)
	# 	evaluation_trajectories_b = preference_model_trajectories.evaluate_action_detach(ret_b)
	# 	compare_actions_ab = preference_model_actions.compare_trajectory(ret_a, ret_b).item()
	# 	compare_actions_ba = preference_model_actions.compare_trajectory(ret_b,ret_a).item()
	# 	compare_trajectories_ab = preference_model_trajectories.compare_trajectory(ret_a, ret_b).item()
	# 	compare_trajectories_ba = preference_model_trajectories.compare_trajectory(ret_b,ret_a).item()
	# 	print("ret_a = ",ret_a)
	# 	print("ret_b = ",ret_b)
	# 	print("evaluation_actions_a = ", evaluation_actions_a)
	# 	print("evaluation_trajectories_a = ", evaluation_trajectories_a)
	# 	print("evaluation_actions_b = ", evaluation_actions_b)
	# 	print("evaluation_trajectories_b = ", evaluation_trajectories_b)
	# 	print("compare_actions_ab = ", compare_actions_ab)
	# 	print("compare_actions_ba = ", compare_actions_ba)
	# 	print("compare_trajectories_ab = ", compare_trajectories_ab)
	# 	print("compare_trajectories_ba = ", compare_trajectories_ba)


	# order = [3,1,0,2]
	# preference_giver = EthicalParetoGiverv3_ObjectiveOrder(order)
	# preference_buffer = PreferenceBufferTest()

	# n_queries = 10
	# for i in range(n_queries):
	# 	ret_a = randomlist = random.sample(range(0, 13), 3) + [random.randint(-8, 0)]
	# 	ret_b = randomlist = random.sample(range(0, 13), 3) + [random.randint(-8, 0)]
	# 	auto_preference = preference_giver.query_pair(ret_a, ret_b)
	# 	preference_buffer.add_preference(ret_a, ret_b, auto_preference)



	# w = [0.5, 0.5, 0.5]
	# nb_samples = 10000
	# # gaussian_samples = np.random.normal(w, 1, nb_samples)
	# cov = [[1 for i in range(len(w))] for j in range(len(w))]
	# cov = np.ones((len(w), len(w)))
	# print(cov)
	# # cov = np.eye(len(w))
	# gaussian_samples = np.random.multivariate_normal(w, cov, nb_samples)
	# moral_samples = st.multivariate_normal(mean=w, cov=0.5).rvs(size=nb_samples)

	# gaussian_mean = np.mean(gaussian_samples, axis=0)
	# moral_mean = np.mean(moral_samples,  axis=0)
	# gaussian_std = np.std(gaussian_samples, axis=0)
	# moral_std = np.std(moral_samples,  axis=0)

	# nb_g_positive = 0
	# nb_m_positive = 0
	# nb_g_len1 = 0
	# nb_m_len1 = 0
	# nb_g_pos_len1 = 0
	# nb_m_pos_len1 = 0
	# for i in range(nb_samples):
	# 	if (gaussian_samples[i]>0).all():
	# 		nb_g_positive += 1
	# 		if np.linalg.norm(gaussian_samples[i])<=1:
	# 			nb_g_pos_len1 += 1
	# 	if (moral_samples[i]>0).all():
	# 		nb_m_positive += 1
	# 		if np.linalg.norm(moral_samples[i])<=1:
	# 			nb_m_pos_len1 += 1
	# 	if np.linalg.norm(gaussian_samples[i])<=1:
	# 		nb_g_len1 += 1
	# 	if np.linalg.norm(moral_samples[i])<=1:
	# 		nb_m_len1 += 1

	# print("gaussian_mean = ", gaussian_mean)
	# print("moral_mean = ", moral_mean)
	# print("gaussian_std = ", gaussian_std)
	# print("moral_std = ", moral_std)
	# print("nb_g_positive = ", nb_g_positive)
	# print("nb_m_positive = ", nb_m_positive)
	# print("nb_g_len1 = ", nb_g_len1)
	# print("nb_m_len1 = ", nb_m_len1)
	# print("nb_g_pos_len1 = ", nb_g_pos_len1)
	# print("nb_m_pos_len1 = ", nb_m_pos_len1)

	# for i in range(len(w)):
	# 	plt.hist(np.histogram(gaussian_samples[i]), bins='auto')
	# 	plt.show()
	# 	plt.hist(np.histogram(moral_samples[i]), bins='auto')
	# 	plt.show()


	# q = st.multivariate_normal(mean=w1, cov=1).pdf(w2)

	# for i in range(10000):
	# 	print(i)
	# 	if i%2==0:
	# 		time.sleep(30)

	ret_a = np.array([1,2,3,-4])
	# ret_b = np.array([0,0,0,0])
	# ret_b = np.array([0,0,4,0])
	ret_b = np.array([1,5,1,0])
	ratio_norm = [0.2, 0.6, 0.2]
	dim = 3
	a = query_pair_no_null(ret_a, ret_b, dim, ratio_norm)
	print(a)