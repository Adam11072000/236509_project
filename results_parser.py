import json
import matplotlib.pyplot as plt
import os
import numpy as np
import copy

OUTPUT_FIGS_DIR = "./output_figures"

def plot_distribution_fault_models(data: dict, fault_model: str, distribution_target: str, sort_key: str):
    fig, axs = plt.subplots(2, figsize=(10,10))
    
    aggregated_avg_per_axis = []
    total_avg = 0
    total_len = 0
    overall_avg = []
    for ax, bit_flips_distrib in zip(axs, ("gaussian", "uniform")):
        for fault_place_in_nn, workers_data in data.items():
            workers = [worker for worker in workers_data if worker[distribution_target] == bit_flips_distrib]
            sorted_list_of_dicts = sorted(workers, key=lambda x: x[sort_key])
            avg_values = [np.mean(worker_data["acc_per_iteration"]) for worker_data in sorted_list_of_dicts]
            aggregated_avg_per_axis.append(avg_values)
            x = range(len(sorted_list_of_dicts))
            ax.plot(x, avg_values, label=fault_place_in_nn)
        for avg in aggregated_avg_per_axis:
            total_len += len(avg)
            total_avg += (len(avg) * np.mean(avg))
        ax.set_title(bit_flips_distrib + "_" + str(total_avg / total_len))
        overall_avg.append({
            "avg": total_avg,
            "len": total_len
        })
        aggregated_avg_per_axis = []
    total_len = 0
    total_avg = 0
    for avg in overall_avg:
        total_avg += avg["avg"]
        total_len += avg["len"]
    total_avg /= total_len
    fig.suptitle(fault_model + f"_{total_avg}")
    for ax in axs:
        ax.legend()
        ax.grid(True)
        ax.set_xlabel("Fault")
        ax.set_ylabel("Accuracy")
    plt.savefig(f"./output_figures/{fault_model}")


def plot_results(data: dict, fault_model: str):
    fig, ax = plt.subplots(1, figsize=(10,10))
    
    total_avg = []
    for fault_place_in_nn, workers_data in data.items():
        sorted_list_of_dicts = sorted(workers_data, key=lambda x: x["num_faults"])
        avg_values = [np.mean(worker_data["acc_per_iteration"]) for worker_data in sorted_list_of_dicts]
        total_avg.append(np.mean(avg_values))
        x = range(len(avg_values))
        ax.plot(x, avg_values, label=fault_place_in_nn)
    total_avg = np.mean(total_avg)
    fig.suptitle(fault_model + f"_{total_avg}")
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Faults")
    ax.set_ylabel("Accuracy")
    # Show the figure
    plt.savefig(f"./output_figures/{fault_model}")

dirs = [
    "distribution_bit_flips_20",
    "distribution_faults",
    "fault_number",
    #"fault_number_randomized_bits",
    "high_bits_exponent",
    "high_bits_mantissa",
    "middle_bits_exponent",
    "middle_bits_mantissa",
    "low_bits_exponent",
    "low_bits_mantissa",
    "sign_bit"
]


if __name__ == "__main__":
    fault_place = {
        "early": [],
        "early_middle": [],
        "late_middle": [],
        "end": []
    }
    fault_types = {
        "weight": copy.deepcopy(fault_place),
        "bias": copy.deepcopy(fault_place),
        "memory": copy.deepcopy(fault_place)
    }
    place_mappings = {
        "bias": {
            "early": "bn1",
            "early_middle": "layer2.1.bn1",
            "late_middle": "layer3.1.bn1",
            "end": "layer4.1.bn1"
        },
        "memory": {
            "early": "conv1",
            "early_middle": "layer2.1.conv2",
            "late_middle": "layer3.1.conv2",
            "end": "layer4.1.conv2"
        },
        "weight": {
            "early": "conv1",
            "early_middle": "layer2.1.conv2",
            "late_middle": "layer3.1.conv2",
            "end": "layer4.1.conv2"
        }
    }
    faults = {}
    for dir_type in dirs:
        faults[dir_type] = copy.deepcopy(fault_types)
    for generic_fault_model in faults:
        for fault_target in faults[generic_fault_model]:
            for fault_place_in_nn in faults[generic_fault_model][fault_target]:
                for distribution in ("gaussian", "uniform"):
                    fault_model = generic_fault_model + "_" + fault_target + "_" + place_mappings[fault_target][fault_place_in_nn]
                    fault_model += f"_{distribution}"
                    try:
                        for data in os.listdir(f"./{fault_model}"):
                            with open(os.path.join(fault_model, data), "r") as file_descriptor:
                                worker_data = json.load(file_descriptor)
                                file_descriptor.close()        
                            faults[generic_fault_model][fault_target][fault_place_in_nn].append(worker_data)
                    except:
                        pass
    
    for generic_fault_model in faults:
        for fault_target in faults[generic_fault_model]:
            fault_model = generic_fault_model + "_" + fault_target
            if generic_fault_model == "distribution_bit_flips_20":
                plot_distribution_fault_models(faults[generic_fault_model][fault_target], fault_model, "fault_distrubution_per_target", "num_bits_to_flip")
            elif generic_fault_model == "distribution_faults":
                plot_distribution_fault_models(faults[generic_fault_model][fault_target], fault_model, "fault_distribution", "num_faults")
            else:
                plot_results(faults[generic_fault_model][fault_target], fault_model)