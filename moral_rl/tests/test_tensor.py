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
import random
from drlhp.preference_model import *

if __name__ == '__main__':


	env_dim = 4
	model_actions = "generated_data/v3/pref_model/ALLCOMBI_5b_2000e_[3, 1, 0, 2].pt"
	preference_model_actions = PreferenceModelTEST(env_dim).to(device)
	preference_model_actions.load_state_dict(torch.load(model_actions, map_location=torch.device('cpu')))
	model_trajectories = "generated_data/v3/pref_model/trajectories/ALLCOMBI_100q_5b_2000e_[3, 1, 0, 2].pt"
	preference_model_trajectories = PreferenceModelTEST(env_dim).to(device)
	preference_model_trajectories.load_state_dict(torch.load(model_trajectories, map_location=torch.device('cpu')))

	n_queries = 10
	for i in range(n_queries):
		ret_a = randomlist = random.sample(range(0, 13), 3) + [random.randint(-8, 0)]
		ret_b = randomlist = random.sample(range(0, 13), 3) + [random.randint(-8, 0)]
		evaluation_actions_a = preference_model_actions.evaluate_action_detach(ret_a)
		evaluation_trajectories_a = preference_model_trajectories.evaluate_action_detach(ret_a)
		evaluation_actions_b = preference_model_actions.evaluate_action_detach(ret_b)
		evaluation_trajectories_b = preference_model_trajectories.evaluate_action_detach(ret_b)
		compare_actions_ab = preference_model_actions.compare_trajectory(ret_a, ret_b).item()
		compare_actions_ba = preference_model_actions.compare_trajectory(ret_b,ret_a).item()
		compare_trajectories_ab = preference_model_trajectories.compare_trajectory(ret_a, ret_b).item()
		compare_trajectories_ba = preference_model_trajectories.compare_trajectory(ret_b,ret_a).item()
		print("ret_a = ",ret_a)
		print("ret_b = ",ret_b)
		print("evaluation_actions_a = ", evaluation_actions_a)
		print("evaluation_trajectories_a = ", evaluation_trajectories_a)
		print("evaluation_actions_b = ", evaluation_actions_b)
		print("evaluation_trajectories_b = ", evaluation_trajectories_b)
		print("compare_actions_ab = ", compare_actions_ab)
		print("compare_actions_ba = ", compare_actions_ba)
		print("compare_trajectories_ab = ", compare_trajectories_ab)
		print("compare_trajectories_ba = ", compare_trajectories_ba)