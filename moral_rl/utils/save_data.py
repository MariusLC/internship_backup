import os 
import torch
import pickle


def save_data(data, filename):
    # print(filename)
    path = filename.split("/")
    # print(path)
    path = path[:-1]
    # print(path)
    path = "/".join(path)
    # print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(data.state_dict(), filename)

def save_demos(dataset, demos_filename):
    # print(filename)
    path = demos_filename.split("/")
    # print(path)
    path = path[:-1]
    # print(path)
    path = "/".join(path)
    # print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    pickle.dump(dataset, open(demos_filename, 'wb'))