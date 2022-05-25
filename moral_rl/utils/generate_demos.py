from envs.gym_wrapper import *
from utils.load_config import *
from utils.generate_demos_not_main import *


# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_GENE_DEMO.yaml"

if __name__ == '__main__':

    c = load_config(CONFIG_PATH, CONFIG_FILENAME)

    vanilla_path = ""
    if c["vanilla"]:
        vanilla_path = c["vanilla_path"]

    for i in range(c["nb_experts"]):
        path = c["data_path"]+c["env_path"]+vanilla_path+str(c["experts_weights"][i])+"/"
        
        expert_filename = path+c["expe_path"]+c["model_ext"]
        demos_filename = path+c["demo_path"]+c["demo_ext"]
        generate_demos_1_expert(c["env_rad"]+c["env"], c["nb_demos"], expert_filename, demos_filename)

        # gene_filename = path+c["gene_path"]+c["model_ext"]
        # demos_filename = path+c["demo_path"]+c["gene"]+c["demo_ext"]
        # # print(expert_filename)
        # generate_demos_1_expert(c["env_rad"]+c["env"], c["nb_demos"], gene_filename, demos_filename)