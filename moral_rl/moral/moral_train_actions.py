from envs.gym_wrapper import *
from utils.load_config import *
from moral.moral_train_not_main_actions import *
from moral.airl import *


# folder to load config file
CONFIG_PATH = "configs/"
CONFIG_FILENAME = "config_MORAL.yaml"

if __name__ == '__main__':

    c = load_config(CONFIG_PATH, CONFIG_FILENAME)

    vanilla_path = ""
    if c["vanilla"]:
        vanilla_path = c["vanilla_path"]

    # will impact the utopia point calculated
    gene_or_expert = c["gene_path"]
    if c["geneORexpert"]:
        gene_or_expert = c["expe_path"]

    query_freq = c["query_freq"]
    if c["real_params"]:
        env_steps = int(c["env_steps"]/c["n_workers"])
        query_freq = int(env_steps/(c["n_queries"]+2))
        # print("env_steps = ", env_steps)
        # print("query_freq = ", query_freq)

    gene_or_expe_filenames = []
    demos_filenames = []
    discriminators_filenames = []
    moral_filename = c["data_path"]+c["env"]+"/"+vanilla_path+c["moral_path"]+str(c["experts_weights"])+c["special_name_agent"]+c["model_ext"]
    non_eth_expert = c["data_path"]+c["env"]+"/"+vanilla_path+str(c["non_eth_experts_weights"])+"/"+c["expe_path"]+c["model_ext"]
    for i in range(c["nb_experts"]):
        path = c["data_path"]+c["env"]+"/"+vanilla_path+str(c["experts_weights"][i])+"/"
        gene_or_expe_filenames.append(path+gene_or_expert+c["model_ext"])
        demos_filenames.append(path+c["demo_path"]+c["demo_ext"])
        discriminators_filenames.append(path+c["disc_path"]+c["model_ext"])

    moral_train_n_experts(c, query_freq, env_steps, gene_or_expe_filenames, discriminators_filenames, moral_filename, non_eth_expert)