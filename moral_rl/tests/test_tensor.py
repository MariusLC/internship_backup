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

import numpy as np
import math
import scipy.stats as st

from moral.active_learning import *
from moral.preference_giver import *

import math

from torch.distributions import *


if __name__ == '__main__':


	# a = torch.tensor(np.array([math.nan, math.nan, math.nan]))
	# print(a)
	# if math.isnan(a[0]):
	# 		print("there is a nan value in result of forward in evaluate_trajectory")

	nb_samples = 1000

	non_eth = Bernoulli(torch.tensor([0.1]))
	rew_obj_non_eth = np.array([non_eth.sample() for i in range(nb_samples)])
	m_non_eth = np.mean(rew_obj_non_eth)
	min_non_eth = min(rew_obj_non_eth)
	max_non_eth = max(rew_obj_non_eth)
	norm1_non_eth = (rew_obj_non_eth - min_non_eth)/(max_non_eth - min_non_eth)
	m_norm1_non_eth = np.mean(norm1_non_eth)
	std_norm1_non_eth = np.std(norm1_non_eth)
	norm2_non_eth = norm1_non_eth/abs(m_non_eth)
	m_norm2_non_eth = np.mean(norm2_non_eth)
	norm3_non_eth = norm1_non_eth-m_norm1_non_eth/std_norm1_non_eth
	m_norm3_non_eth = np.mean(norm3_non_eth)
	norm4_non_eth = rew_obj_non_eth / abs(m_non_eth)
	m_norm4_non_eth = np.mean(norm4_non_eth)

	eth = Normal(torch.tensor([-2.0]), torch.tensor([2.0]))
	rew_obj_eth = np.array([eth.sample() for i in range(nb_samples)])
	m_eth = np.mean(rew_obj_eth)
	min_eth = min(rew_obj_eth)
	max_eth = max(rew_obj_eth)
	norm1_eth = (rew_obj_eth - min_eth)/(max_eth - min_eth)
	m_norm1_eth = np.mean(norm1_eth)
	std_norm1_eth = np.std(norm1_eth)
	norm2_eth = norm1_eth/abs(m_norm1_eth)
	m_norm2_eth = np.mean(norm2_eth)
	norm3_eth = norm1_eth-m_norm1_eth/std_norm1_eth
	m_norm3_eth = np.mean(norm3_eth)
	norm4_eth = rew_obj_eth / abs(m_eth)
	m_norm4_eth = np.mean(norm4_eth)

	# print("rew_obj_eth = ", rew_obj_eth)
	# print("norm1_eth = ", norm1_eth)
	# print("norm2_eth = ", norm2_eth)
	# print("norm3_eth = ", norm3_eth)
	print("m_eth = ", m_eth)
	print("m_norm1_eth = ", m_norm1_eth)
	print("m_norm2_eth = ", m_norm2_eth)
	print("m_norm3_eth = ", m_norm3_eth)
	print("m_norm4_eth = ", m_norm4_eth)

	# print("\n\rew_obj_non_eth = ", rew_obj_non_eth)
	# print("norm1_non_eth = ", norm1_non_eth)
	# print("norm2_non_eth = ", norm2_non_eth)
	# print("norm3_non_eth = ", norm3_non_eth)
	print("m_non_eth = ", m_non_eth)
	print("m_norm1_non_eth = ", m_norm1_non_eth)
	print("m_norm2_non_eth = ", m_norm2_non_eth)
	print("m_norm3_non_eth = ", m_norm3_non_eth)
	print("m_norm4_non_eth = ", m_norm4_non_eth)