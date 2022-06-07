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


if __name__ == '__main__':


	a = torch.tensor(np.array([math.nan, math.nan, math.nan]))
	print(a)
	if math.isnan(a[0]):
            print("there is a nan value in result of forward in evaluate_trajectory")
