from envs.gym_wrapper import *
from utils.load_config import *
from moral.ppo_train_not_main import *


# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_PPO.yaml"

if __name__ == '__main__':

    c = load_config(CONFIG_PATH, CONFIG_FILENAME)

    vanilla_path = ""
    if c["vanilla"]:
        vanilla_path = c["vanilla_path"]

    expert_filename = c["data_path"]+c["env_path"]+vanilla_path+str(c["experts_weights"])+"/"+c["expe_path"]+c["special_name"]+c["model_ext"]
    # print(expert_filename)
    ppo_train_1_expert(c, expert_filename)
