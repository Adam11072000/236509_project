import json
import matplotlib.pyplot as plt
import os
import numpy as np
import copy

OUTPUT_FIGS_DIR = "./output_figures"

def plot_results(data: dict, name: str):
    fig, axs = plt.subplots(2, figsize=(10,10))
    fig.suptitle(f"{generic_fault_model}_{fault_target}")
    for fault_place_in_nn, distributions in data.items():
        for ax, (distribution, workers_data) in zip(axs, distributions.items()):
            avg_values = [np.mean(worker_data["acc_per_iteration"]) for worker_data in workers_data]
            x = range(len(avg_values))
            ax.plot(x, avg_values, label=fault_place_in_nn)
            ax.set_title(distribution)
    for ax in axs:
        ax.legend()
        ax.grid(True)
    # Show the figure
    plt.savefig(f"./output_figures/{name}")

dirs = [
    "distribution_bit_flips_20",
    "distribution_bit_flips_100",
    "distribution_bit_flips_60",
    "distribution_faults",
    "fault_number",
    "fault_number_randomized_bits",
    "high_bits",
    "middle_bits",
    "low_bits"
]


if __name__ == "__main__":
    distributions = {
        "uniform":[],
        "gaussian":[]
    }
    fault_place = {
        "early": copy.deepcopy(distributions),
        "early_middle": copy.deepcopy(distributions),
        "late_middle": copy.deepcopy(distributions),
        "end": copy.deepcopy(distributions)
    }
    fault_types = {
        "weight": copy.deepcopy(fault_place),
        "bias": copy.deepcopy(fault_place),
        "memory": copy.deepcopy(fault_place)
    }
    place_mappings = {
        "bias": {
            "early": "bn1",
            "early_middle": "layer3.0.bn1",
            "late_middle": "layer3.1.bn1",
            "end": "layer4.1.bn2"
        },
        "memory": {
            "early": "conv1",
            "early_middle": "layer3.0.conv2",
            "late_middle": "layer4.0.conv1",
            "end": "layer4.1.conv2"
        },
        "weight": {
            "early": "conv1",
            "early_middle": "layer3.0.conv2",
            "late_middle": "layer4.0.conv1",
            "end": "layer4.1.conv2"
        }
    }
    faults = {}
    for dir_type in dirs:
        faults[dir_type] = copy.deepcopy(fault_types)
    for generic_fault_model in faults:
        for fault_target in faults[generic_fault_model]:
            for fault_place_in_nn in faults[generic_fault_model][fault_target]:
                fault_model = generic_fault_model + "_" + fault_target + "_" + place_mappings[fault_target][fault_place_in_nn]
                for data in os.listdir(f"./{fault_model}"):
                    with open(os.path.join(fault_model, data), "r") as file_descriptor:
                        worker_data = json.load(file_descriptor)
                        file_descriptor.close()
                    if "fault_distribution" in worker_data:
                        fault_distribution = worker_data["fault_distribution"]
                    else:
                        fault_distribution = worker_data[list(worker_data.keys())[0]]["fault_distribution"]
                        worker_data = worker_data[list(worker_data.keys())[0]]
                    faults[generic_fault_model][fault_target][fault_place_in_nn][fault_distribution].append(worker_data)
    
    for generic_fault_model in faults:
        for fault_target in faults[generic_fault_model]:
            fault_model = generic_fault_model + "_" + fault_target
            plot_results(faults[generic_fault_model][fault_target], fault_model)
