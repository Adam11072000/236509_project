import json
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_results(data: dict, name: str, sorting_key):
    plt.figure(figsize=(10,6))

    x = range(1, len(data) + 1)
    data = sorted(data, key=lambda x: x[sorting_key])
    max_data = []
    min_data = []
    avg = []
    for fault_model in data:
        max_data.append(max(fault_model["acc_per_iteration"]))
        min_data.append(min(fault_model["acc_per_iteration"]))
        avg.append(np.mean(fault_model["acc_per_iteration"]))

    plt.plot(x, max_data, label='max over fault models')
    plt.plot(x, min_data, label='min over fault models')
    plt.plot(x, avg, label='avg over fault models')

    plt.title("%s, original accuracy: %s" % (name, data[0]["original_acc"]))
    plt.xlabel("Fault")
    plt.ylabel("Accuracy")

    plt.grid()
    plt.legend()

    name = name.replace(" ", "_")
    plt.savefig(f'{name}.png')


def parse_results(target_dir: str, target_key):
    # each dir has the main metric, but also faults locations are distributed via 2 different distributions, uniform and gaussian
    faults = {}
    for file in os.listdir(target_dir):
        with open(os.path.join(target_dir, file), "r") as file_descriptor:
            worker_data = json.load(file_descriptor)
            file_descriptor.close()
        fault_distribution = worker_data[list(worker_data.keys())[0]]["fault_distribution"]
        if not fault_distribution in faults:
            faults[fault_distribution] = []
        faults[fault_distribution].append(worker_data[list(worker_data.keys())[0]])
    
    plot_results(faults["uniform"], f"Uniform distrubuted faults over {target_dir}", target_key)
    plot_results(faults["gaussian"], f"Gaussian distrubuted faults over {target_dir}", target_key)



if __name__ == "__main__":
    for entry in os.listdir("."):
        if (entry.find("_bias") != -1 or entry.find("_memory") != -1 or entry.find("_weight") != -1) and os.path.isdir(entry):
            if entry.find("bit_flips") != -1:
                target_key = "num_bits_to_flip"
            else:
                target_key = "num_faults"
            parse_results(entry, target_key)
    