import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from envs.gym_wrapper import *

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
import yaml
import os
import math

import numpy as np
import math
import scipy.stats as st


# folder to load config file
CONFIG_PATH = "moral/"
CONFIG_FILENAME = "config_MORAL.yaml"

# Function to load yaml configuration file
def load_config(config_name):
	with open(os.path.join(CONFIG_PATH, config_name)) as file:
		config = yaml.safe_load(file)

	return config

D = 3

def w_prior(w):
	if np.linalg.norm(w) <=1 and np.all(np.array(w) >= 0):
		return (2**D)/(math.pi**(D/2)/math.gamma(D/2 + 1))
	else:
		return 0

# @staticmethod
def f_loglik(w, delta, pref):
	return np.log(np.minimum(1, np.exp(pref*np.dot(w, delta)) + 1e-5))

# @staticmethod
def vanilla_loglik(w, delta, pref):
	return np.log(1/(1+np.exp(-pref*np.dot(w, delta))))

# @staticmethod
def propose_w_prob(w1, w2):
	q = st.multivariate_normal(mean=w1, cov=1).pdf(w2)
	return q

# @staticmethod
def propose_w(w_curr):
	w_new = st.multivariate_normal(mean=w_curr, cov=1).rvs()
	return w_new

def posterior_log_prob(deltas, prefs, w):
	f_logliks = []
	for i in range(len(prefs)):
		f_logliks.append(f_loglik(w, deltas[i], prefs[i]))
	loglik = np.sum(f_logliks)
	log_prior = np.log(w_prior(w) + 1e-5)

	return loglik + log_prior

def mcmc_vanilla(deltas, prefs, warmup, n_iter, w_init='mode'):
		if w_init == 'mode':
			w_init = [0 for i in range(D)]

		w_arr = []
		w_curr = w_init
		accept_rates = []
		accept_cum = 0

		for i in range(1, warmup + n_iter + 1):
			w_new = propose_w(w_curr)
			

			prob_curr = posterior_log_prob(deltas, prefs, w_curr)
			
			prob_new = posterior_log_prob(deltas, prefs, w_new)
			
			if prob_new > prob_curr:
				print("w_new = ", w_new)
				print("prob_curr = ", prob_curr)
				print("prob_new = ", prob_new)
				acceptance_ratio = 1
			else:
				qr = propose_w_prob(w_curr, w_new) / propose_w_prob(w_new, w_curr)
				acceptance_ratio = np.exp(prob_new - prob_curr) * qr
				print("acceptance_ratio = ", acceptance_ratio)
				print("qr = ", qr)
				print("np.exp(prob_new - prob_curr) = ", np.exp(prob_new - prob_curr))
			acceptance_prob = min(1, acceptance_ratio)

			if acceptance_prob > st.uniform(0, 1).rvs():
				w_curr = w_new
				accept_cum = accept_cum + 1
				# print(w_new)
				w_arr.append(w_new)
			else:
				# print("w_curr = ", w_curr)
				w_arr.append(w_curr)

			accept_rates.append(accept_cum / i)

		accept_rates = np.array(accept_rates)[warmup:]
		return np.array(w_arr)[warmup:]

def f_loglik_print(w, delta, pref):
	print(delta)
	temp = []
	for i in range(len(delta)):
		print("temps = ",w[i]*delta[i])
		temp.append(w[i]*delta[i])
	print(np.array(temp)*pref)
	return np.log(np.minimum(1, np.exp(pref*np.dot(w, delta)) + 1e-5))

if __name__ == '__main__':

	# preference_giver = PreferenceGiverv3(ratio=[1,3,1])

	deltas = np.array([
		[  0.  ,        80.24958384 ,-77.16231537],
		[   0.   ,      -125.59433675 ,  42.68571472],
		[ -3.    ,      81.9175384 , -48.91356659],
		[  1.     ,    103.90283465 ,-54.73291779],
		[ -2.    ,      69.03929481 ,-36.77306366],
		[ -1.     ,     57.22481468 ,-33.91828156],
		[  2.    ,      81.65642965 ,-52.90368652],
		[  0.     ,    -79.31589961 , 48.51434326],
		[   5.    ,       -3.78819871 ,-127.3263855 ]
	])

	prefs = [
		-1,
		-1,
		-1,
		1,
		1,
		1,
		-1,
		-1,
		1
	]
	d=len([1,3,1]) # dim ratio lambd
	n_iter=10000
	warmup=1000

	# w_posterior = np.zeros(3)
	w_posterior_mean = [0, 0, 0]
	i = 1


	for i in range(len(deltas)):
		# a = f_loglik_print(w_posterior_mean, deltas[i], prefs[i])
		# print("a = ", a)
		w_new = propose_w(w_posterior_mean)
		b = f_loglik_print(w_new, deltas[i], prefs[i])
		print("b = ", b)

	# while(not math.isnan(w_posterior_mean[0])):
	# 	# preference = preference_giver.query_pair(ret_a, ret_b)


	# 	# w_posterior = mcmc_vanilla(deltas[:i], prefs[:i], warmup, n_iter)
	# 	# w_posterior = mcmc_vanilla(deltas[-1:], prefs[-1:], warmup, n_iter)
	# 	w_posterior = mcmc_vanilla(deltas, prefs, warmup, n_iter)
	# 	print("w_posterior = ", str(w_posterior))
	# 	w_posterior_mean = w_posterior.mean(axis=0)
	# 	print("w_posterior_mean = ", str(w_posterior_mean))
	# 	# making a 1 norm vector from w_posterior
	# 	w_posterior_mean = w_posterior_mean/np.linalg.norm(w_posterior_mean)
	# 	print(f'Posterior Mean {w_posterior_mean}')
	# 	# print(np.log(w_prior(np.array([0., 0., 0.])) + 1e-5))
	# 	if i < len(deltas):
	# 		i += 1

	