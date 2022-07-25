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

	expert_filename = c["agent_name"]

	# experts
	expert_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
	expert_policy.load_state_dict(torch.load(expert_filename, map_location=torch.device('cpu')))

	# rand agents
	rand_policy = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)

	if c["generate_demos"] :
			generate_demos_1_expert(c["env_rad"]+c["env"], c["nb_demos"], expert_filename, demos_filename)
			# generate_demos_1_expert(c["env_rad"]+c["env"], c["nb_demos"], rand_filename, rand_demos_filename)

	# evaluate agents
	res_exp, std_exp = evaluate_ppo_2(expert_policy, c["env_id"], c["nb_steps"])
	res_rand, std_rand = evaluate_ppo_2(rand_policy, c["env_id"], c["nb_steps"])
	app_res_exp = [round(r, 2) for r in res_exp]
	app_res_rand = [round(r, 2) for r in res_rand]
	print("eval expert = ", app_res_exp)
	print("eval rand = ", app_res_rand)