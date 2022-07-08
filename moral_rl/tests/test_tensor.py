# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
# from envs.gym_wrapper import *

# from tqdm import tqdm
import torch
# import matplotlib.pyplot as plt
# import wandb
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

if __name__ == '__main__':


	# a = torch.tensor(np.array([math.nan, math.nan, math.nan]))
	# print(a)
	# if math.isnan(a[0]):
	# 		print("there is a nan value in result of forward in evaluate_trajectory")

	# nb_samples = 1000

	# non_eth = Bernoulli(torch.tensor([0.1]))
	# rew_obj_non_eth = np.array([non_eth.sample() for i in range(nb_samples)])
	# m_non_eth = np.mean(rew_obj_non_eth)
	# min_non_eth = min(rew_obj_non_eth)
	# max_non_eth = max(rew_obj_non_eth)
	# norm1_non_eth = (rew_obj_non_eth - min_non_eth)/(max_non_eth - min_non_eth)
	# m_norm1_non_eth = np.mean(norm1_non_eth)
	# std_norm1_non_eth = np.std(norm1_non_eth)
	# norm2_non_eth = norm1_non_eth/abs(m_non_eth)
	# m_norm2_non_eth = np.mean(norm2_non_eth)
	# norm3_non_eth = norm1_non_eth-m_norm1_non_eth/std_norm1_non_eth
	# m_norm3_non_eth = np.mean(norm3_non_eth)
	# norm4_non_eth = rew_obj_non_eth / abs(m_non_eth)
	# m_norm4_non_eth = np.mean(norm4_non_eth)

	# traj_size = 73
	# nb_traj = 10
	# eth = Normal(torch.tensor([-2.0]), torch.tensor([2.0]))
	# rew_obj_eth = np.array([eth.sample() for i in range(traj_size*nb_traj)])
	# mean_per_traj = [np.sum(rew_obj_eth[a*traj_size:(a+1)*traj_size]) for a in range(nb_traj)]
	# up = sum(rew_obj_eth)/nb_traj
	# min_traj = min(mean_per_traj)
	# max_traj = max(mean_per_traj)
	# lower_bound = min(rew_obj_eth)
	# upper_bound = max(rew_obj_eth)
	# norm_up = (up - traj_size*lower_bound)/(upper_bound - lower_bound)
	# norm_v2 = (rew_obj_eth - lower_bound)/(upper_bound - lower_bound)/abs(norm_up)
	# norm_v3 = (rew_obj_eth - min_traj/traj_size)/(max_traj - min_traj)
	# m_norm_v2 = [np.sum(norm_v2[a*traj_size:(a+1)*traj_size]) for a in range(nb_traj)]
	# m_norm_v3 = [np.sum(norm_v3[a*traj_size:(a+1)*traj_size]) for a in range(nb_traj)]

	# print("min_traj = ", min_traj)
	# print("max_traj = ", max_traj)
	# print("lower_bound = ", lower_bound)
	# print("upper_bound = ", upper_bound)
	# print("up = ", up)
	# print("norm_up = ", norm_up)
	# print("m_norm_v2 = ", m_norm_v2)
	# print("m_norm_v3 = ", m_norm_v3)
	# print_val = [(rew_obj_eth[i], norm_v2[i], norm_v3[i]) for i in range(len(rew_obj_eth))]
	# print(print_val)

	# m_eth = np.mean(rew_obj_eth)
	# min_eth = min(rew_obj_eth)
	# max_eth = max(rew_obj_eth)
	# norm1_eth = (rew_obj_eth - min_eth)/(max_eth - min_eth)
	# m_norm1_eth = np.mean(norm1_eth)
	# m_norm1_eth_test = (m_eth - min_eth)/(max_eth - min_eth)
	# std_norm1_eth = np.std(norm1_eth)
	# norm2_eth = norm1_eth/abs(m_norm1_eth)
	# m_norm2_eth = np.mean(norm2_eth)
	# m_norm2_eth_test = m_norm1_eth_test/abs(m_norm1_eth_test)
	# norm3_eth = norm1_eth-m_norm1_eth/std_norm1_eth
	# m_norm3_eth = np.mean(norm3_eth)
	# norm4_eth = rew_obj_eth / abs(m_eth)
	# m_norm4_eth = np.mean(norm4_eth)

	# # print("rew_obj_eth = ", rew_obj_eth)
	# # print("norm1_eth = ", norm1_eth)
	# # print("norm2_eth = ", norm2_eth)
	# # print("norm3_eth = ", norm3_eth)
	# print("m_eth = ", m_eth)
	# print("m_norm1_eth = ", m_norm1_eth)
	# print("m_norm1_eth_test = ", m_norm1_eth_test)
	# print("m_norm2_eth = ", m_norm2_eth)
	# print("m_norm2_eth_test = ", m_norm2_eth_test)
	# print("m_norm3_eth = ", m_norm3_eth)
	# print("m_norm4_eth = ", m_norm4_eth)

	# return ((advantage-self.lower_bound)/(self.upper_bound - self.lower_bound))/abs(self.normalized_utopia_point)
 #    return (advantage - self.rewards_min_traj/self.traj_size)/(self.rewards_max_traj - self.rewards_min_traj)

	# # print("\n\rew_obj_non_eth = ", rew_obj_non_eth)
	# # print("norm1_non_eth = ", norm1_non_eth)
	# # print("norm2_non_eth = ", norm2_non_eth)
	# # print("norm3_non_eth = ", norm3_non_eth)
	# print("m_non_eth = ", m_non_eth)
	# print("m_norm1_non_eth = ", m_norm1_non_eth)
	# print("m_norm2_non_eth = ", m_norm2_non_eth)
	# print("m_norm3_non_eth = ", m_norm3_non_eth)
	# print("m_norm4_non_eth = ", m_norm4_non_eth)


	# test = Normal(torch.tensor([-2.5]), torch.tensor([2.0]))
	# rew_obj_test = np.array([test.sample() for i in range(nb_samples)])
	# m_test = np.mean(rew_obj_test)
	# norm1_test = (rew_obj_test - min_eth)/(max_eth - min_eth)
	# m_norm1_test = np.mean(norm1_test)
	# norm2_test = norm1_test/abs(m_eth)
	# m_norm2_test = np.mean(norm2_test)
	# norm2_2_test = norm1_test/abs(m_norm1_eth_test)
	# m_norm2_2_test = np.mean(norm2_2_test)
	# norm2_3_test = (rew_obj_test - min_eth)/(max_eth - min_eth)/abs(m_norm1_eth_test)
	# m_norm2_3_test = np.mean(norm2_3_test)
	# norm2_4_test = (m_test - min_eth)/(max_eth - min_eth)/abs(m_norm1_eth_test)
	# norm2_5_test = (m_test - min_eth)/(max_eth - min_eth)
	# print("m_test = ", m_test)
	# print("m_norm1_test = ", m_norm1_test)
	# print("m_norm2_test = ", m_norm2_test)
	# print("m_norm2_2_test = ", m_norm2_2_test)
	# print("m_norm2_3_test = ", m_norm2_3_test)
	# print("norm2_4_test = ", norm2_4_test)
	# print("norm2_5_test = ", norm2_5_test)


	# a = np.array([-2., -5., 3])
	# a = np.array([0., 0., 0., 1., 1.])
	# b = np.array([1,3,1])
	# print(b)
	# w_posterior_mean = b/np.linalg.norm(b)
	# print(w_posterior_mean)
	# print(np.linalg.norm(w_posterior_mean))


	# up
	# up_a = np.sum(a)
	# a_div_up = a/abs(up_a)
	# m_a_div_up = np.sum(a_div_up)
	# # norm_up
	# min_a = min(a)
	# max_a = max(a)
	# norm_a = (a - min_a)/(max_a-min_a)
	# pre_up = (a/abs(up_a) - min_a)/(max_a-min_a)
	# print("faobeo = ", norm_a/sum(norm_a))
	# print("faobeo = ", pre_up)
	# print("faobeo = ", sum(pre_up))
	# up_norm = (up_a - len(a)*min_a)/(max_a-min_a)
	# print("up_norm = ", up_norm)
	# pos_up = norm_a / up_norm
	# print("pos_up = ", pos_up)
	# m_pos_up = np.sum(pos_up)
	# print("m_pos_up = ", m_pos_up)
	# print("test = ", np.sum(a))

	# norm_a_div_up = norm_a / abs(up_a)
	# m_norm_a_div_up = np.sum(norm_a_div_up)
	# up_a_norm = (up_a - min_a)/(max_a-min_a)
	# norm_a_div_up_a_norm = norm_a / abs(up_a_norm)
	# m_norm_a_div_up_a_norm = np.sum(norm_a_div_up_a_norm)

	# print("up_a = ", up_a)
	# print("a_div_up = ", a_div_up)
	# print("m_a_div_up = ", m_a_div_up)

	# print("norm_a = ", norm_a)
	# print("norm_a_div_up = ", norm_a_div_up)
	# print("m_norm_a_div_up = ", m_norm_a_div_up)
	# print("up_a_norm = ", up_a_norm)
	# print("norm_a_div_up_a_norm = ", norm_a_div_up_a_norm)
	# print("m_norm_a_div_up_a_norm = ", m_norm_a_div_up_a_norm)
	
	# preference_giver = ParetoSoftmaxGiverv3()
	# # preference_giver = EthicalParetoTestGiverv3()
	

	# # Cas 1 : B est fort que pour la tâche primaire, alors que B est moins bon mais respecte beaucoup plus l'ethique.
	# # solution voulue -> on choisit B
	# logs_tau_a = np.array([10, 0, 1, -3])
	# logs_tau_b = np.array([3, 2, 1, 0])
	# print(f'Found trajectory pair: {logs_tau_a, logs_tau_b}')
	# auto_preference = preference_giver.query_pair(logs_tau_a, logs_tau_b)
	# print("auto_preference = ", auto_preference)
	# print("on veut pref = ", str([0, 1]))

	# # Cas 2 : équilibré, A est plus performant que B sur un obj éthique mais B l'est plus sur 2 obj, avec une diff plus faible
	# # solution voulue -> on choisit B ??
	# logs_tau_a = np.array([3, 2, 1, -1])
	# logs_tau_b = np.array([3, 0, 2, 0])
	# print(f'Found trajectory pair: {logs_tau_a, logs_tau_b}')
	# auto_preference = preference_giver.query_pair(logs_tau_a, logs_tau_b)
	# print("auto_preference = ", auto_preference)
	# print("on veut pref = ", str([0, 1]))

	# # Cas 3 : A est un peu moins performant que B sur les obj éthiques mais BEACOUP plus performant sur la tâche primaire.
	# # solution voulue -> on choisit A ?? 
	# # Ou alors on veut que l'éthique prime toujours et ne faire que des compromis entre objectifs l'éthiques, et dans ce cas il faut choisir B.
	# logs_tau_a = np.array([10, 0, 1, -1])
	# logs_tau_b = np.array([0, 1, 1, 0])
	# print(f'Found trajectory pair: {logs_tau_a, logs_tau_b}')
	# auto_preference = preference_giver.query_pair(logs_tau_a, logs_tau_b)
	# print("auto_preference = ", auto_preference)
	# print("on veut pref = ", str([1, 0]))

	all_combi = [[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]
	all_combi = [np.array(c) for c in all_combi]
	a = itertools.product(all_combi, repeat=2)
	# a = itertools.combinations(all_combi, r=2)
	b = list(a)
	c = np.array(b)


	print(a)
	print("\n\n",b)
	print("\n\n",c)
	print(len(b))