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
from utils.generate_demos_not_main import *
from utils.evaluate_ppo_not_main import *
from utils.load_config import *
from moral.airl import *
from utils.save_data import *


# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_EVALUATE_AGENT.yaml"

if __name__ == '__main__':

	c = load_config(CONFIG_PATH, CONFIG_FILENAME)

	# Create Environment
	vec_env = VecEnv(c["env_id"], 12)
	states = vec_env.reset()
	states_tensor = torch.tensor(states).float().to(device)

	# Fetch Shapes
	n_actions = vec_env.action_space.n
	obs_shape = vec_env.observation_space.shape
	state_shape = obs_shape[:-1]
	in_channels = obs_shape[-1]

	if c["mode"] == "agent" :
		expert_filename = c["agent_filename"]
		expert_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
		expert_policy.load_state_dict(torch.load(expert_filename, map_location=torch.device('cpu')))
		mean, std = evaluate_ppo(expert_policy, c["env_id"], c["nb_steps"])

	elif c["mode"] == "demos" :
		demos = pickle.load(open(c["demos_filename"], 'rb'))
		mean, std = evaluate_from_demos(demos)

	mean_roud = [round(r, 2) for r in mean]
	print("eval expert = ", mean_roud)



	# # rand agents
	# rand_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)

	# eval_rand, std_rand = evaluate_ppo(rand_policy, c["env_id"], c["nb_steps"])
	# app_res_rand = [round(r, 2) for r in eval_rand]
	# print("eval rand = ", app_res_rand)