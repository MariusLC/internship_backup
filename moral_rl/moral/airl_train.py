from envs.gym_wrapper import *
from utils.load_config import *
from moral.airl_train_not_main import *


# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_AIRL.yaml"

if __name__ == '__main__':

    c = load_config(CONFIG_PATH, CONFIG_FILENAME)

    vanilla_path = ""
    if c["vanilla"]:
        vanilla_path = c["vanilla_path"]

    path = c["data_path"]+c["env_path"]+vanilla_path+str(c["expert_weights"])+"/"
    expert_filename = path+c["expe_path"]+c["model_ext"]
    demos_filename = path+c["demo_path"]+c["demo_ext"]
    generator_filename = path+c["gene_path"]+c["model_ext"]
    discriminator_filename = path+c["disc_path"]+c["model_ext"]
    # print(demos_filename)
    airl_train_1_expert(c, demos_filename, generator_filename, discriminator_filename, prints=False)


