import torchvision
import torch
from utils import build_module_dict
from FaultInjector import FaultModel, FAULT_TARGET, FAULT_OPTIONS, FAULT_MODE_PER_TARGET, RANDOM_DISTRIBUTIONS, WrongParameters
from scipy.stats import truncnorm
import numpy as np
import subprocess
import argparse
import robustbench


# Add the arguments
parser = argparse.ArgumentParser(description='Arguments for the script. For the list of available models, check https://github.com/RobustBench/robustbench')

parser.add_argument('-n', '--number_of_fault_models', type=int, default=1,
                    help='Number of fault models (must be higher than 0)')
parser.add_argument('-i', '--number_of_iterations_per_fault', type=int, default=1,
                    help='Number of iterations per fault (must be higher than 0)')
parser.add_argument('-o', "--fault_option", type=str, default="non-strict", choices=FAULT_OPTIONS,
                    help="strict or non-strict, i.e., 1/multiple bit flips per fault accordingly.")
parser.add_argument("-m", "--fault_mode_per_target", type=str, default="random", choices=FAULT_MODE_PER_TARGET,
                    help='Fault mode per target (for bit flips), random or deterministic')
parser.add_argument("-t", "--target_bits", nargs='*', default=[], 
                    help="The target bits. Provide multiple values separated by space. Default is an empty list.")
parser.add_argument("-l", "--layer_name", type=str, 
                    help="The name of the layer.")
parser.add_argument("-f", "--fault_target", choices=FAULT_TARGET,
                    help="The target of the fault.")
parser.add_argument("--module", type=str, help="modules available at robustBench, provide them as <threat_model>@<model_name>\
                    threat_model = [Linf, L2, corruptions, corruptions_3d]")
parser.add_argument("--output_dir", type=str, help="Data output directory")
parser.add_argument("--num_faults", type=int, help="The number of faults.")

# Parse the arguments
args = parser.parse_args()

# Ensure number_of_faults, number_of_iterations_per_fault, and num_bits_to_flip are greater than 0
if args.number_of_fault_models < 1:
    parser.error("number_of_faults must be higher than 0")

if args.number_of_iterations_per_fault < 1:
    parser.error("number_of_iterations_per_fault must be higher than 0")

if (args.fault_option == "strict" and len(args.target_bits) > 1) or \
    (args.fault_option == "non-strict" and len(args.target_bits) > 0 and len(args.target_bits) == 1):
    parser.error("strict fault option should have only 1 target bit per fault and non strict should have more than 1 bit flip per target")

if args.fault_mode_per_target == "deterministic" and len(args.target_bits) == 0:
    parser.error("deterministic fault mdoe per target should have pre-chosen bits")


device = "cuda" if torch.cuda.is_available() else "cpu"

def get_supported_layers_per_fault_target(model: torch.nn.Module):
    sub_modules = build_module_dict(model)
    supported_layers_per_fault_target = {
        "weight": [],
        "bias": [],
        "memory": []
    }
    for module_name, module in sub_modules.items():
        if hasattr(module, "weight") and module.weight is not None:
            supported_layers_per_fault_target["weight"].append({"module_name": module_name, "module": module})
        if hasattr(module, "bias") and module.bias is not None:
            supported_layers_per_fault_target["bias"].append({"module_name": module_name, "module": module})
        supported_layers_per_fault_target["memory"].append({"module_name": module_name, "module": module})
    return supported_layers_per_fault_target    


def generate_fault_models(supported_layers_per_fault_target: dict, num_fault_models: int) -> list[FaultModel]: 
    fault_models = []

    # fault target
    lower_bound_fault_num = 0
    upper_bound_fault_num = num_fault_models
    std = 2
    mean = 5
    
    # bit flips
    lower_bound_bit_flips = 2
    upper_bound_bit_flips = 7
    
    # targets
    target_choices = np.random.choice(FAULT_TARGET, size=num_fault_models)
    for i in range(0, num_fault_models):
        # random choosers 
        distribution_target = np.random.choice(RANDOM_DISTRIBUTIONS, size=1)[0]
        distribution_bit_flips = np.random.choice(RANDOM_DISTRIBUTIONS, size=1)[0]
        if distribution_bit_flips == "gaussian":
            a, b = (lower_bound_bit_flips - mean) / std, (upper_bound_bit_flips - mean) / std
            bit_flips = truncnorm.rvs(a, b, loc=mean, scale=std, size=1)
        else:
            bit_flips = np.random.uniform(low=lower_bound_bit_flips, 
                        high=upper_bound_bit_flips, size=1)
        
        # num faults decisions
        if distribution_target == "gaussian":
            a, b = (lower_bound_fault_num - mean) / std, (upper_bound_fault_num - mean) / std
            num_faults = truncnorm.rvs(a, b, loc=mean, scale=std, size=1)
        else:
            num_faults = np.random.uniform(low=lower_bound_fault_num, # need to support gaussian
                                 high=upper_bound_fault_num, size=1)
        num_faults = int(np.around(num_faults, 0)[0])
        if args.num_faults:
            num_faults = args.num_faults

        # bit flips decisions
        bit_flips = int(np.around(bit_flips, 0)[0])
        fault_option = "non-strict" # will do strict later   
        len_target_bits = len(args.target_bits)
        target_bits = []
        if args.fault_option == "strict" and len_target_bits == 1:
            bit_flips = 1
            fault_option = "strict" 
            target_bits = args.target_bits
        if args.fault_option == "non-strict" and len_target_bits > 1:
            bit_flips = len_target_bits
            target_bits = args.target_bits
        target_bits = [int(bit) for bit in target_bits]

        #fault target decisions
        fault_target = target_choices[i]
        if args.fault_target is not None:
            fault_target = args.fault_target

        # layer name decisions
        layer_name = np.random.choice(supported_layers_per_fault_target[fault_target], size=1)[0]["module_name"]
        if args.layer_name is not None:
            layer_name = args.layer_name
            if not any(layer["module_name"] == layer_name for layer in supported_layers_per_fault_target[fault_target]):
                raise WrongParameters("layer_name is not supported with given target")
        
        # creation of fault
        fault_models.append(
            FaultModel(
                layer_name=layer_name,
                fault_target=fault_target,
                num_faults=num_faults,
                fault_option=fault_option,
                num_bits_to_flip=bit_flips,
                fault_distribution=distribution_target,
                fault_mode_per_target="random",
                fault_distrubution_per_target=distribution_bit_flips,
                target_bits=target_bits
            )
        )
    
    return fault_models


if __name__ == "__main__":
    number_of_fault_models = args.number_of_fault_models
    number_of_iterations_per_fault_model = args.number_of_iterations_per_fault

    if args.module:
        model_name = args.module.split("@")[1]
        thread_model = args.module.split("@")[0]
        model = robustbench.utils.load_model(model_name=model_name, threat_model=thread_model, dataset="cifar10")
    else:  
        # use one of my own, for testing purposes.  
        model = torchvision.models.resnet18()
        state_dict = torch.load("model_82.17.pth")
        model.load_state_dict(state_dict)
    model.eval()

    supported_layers_per_fault_target = get_supported_layers_per_fault_target(model)
    fault_models = generate_fault_models(supported_layers_per_fault_target, number_of_fault_models)

    max_workers = 3
    curr_workers = [] # Popen objects
    for i, fault_model in enumerate(fault_models):
        layer_name = fault_model.layer_name
        fault_target = fault_model.fault_target
        num_faults = fault_model.num_faults
        num_bits_to_flip = fault_model.num_bits_to_flip
        fault_distribution = fault_model.fault_distribution
        fault_mode_per_target = fault_model.fault_mode_per_target
        fault_distrubution_per_target = fault_model.fault_distrubution_per_target
        target_bits = ' '.join(str(bit) for bit in fault_model.target_bits)
        worker_id = i
        proc_args = [
            "python3", "worker.py", "--layer_name", layer_name, "--fault_target", fault_target,
            "--num_faults", str(num_faults), "--num_bits_to_flip", str(num_bits_to_flip),
            "--fault_distribution", fault_distribution, "--fault_mode_per_target", fault_mode_per_target,
            "--fault_distrubution_per_target", fault_distrubution_per_target,
            "--number_of_iterations_per_fault_model", str(number_of_iterations_per_fault_model), "--worker_id", str(worker_id)
        ]
        if len(fault_model.target_bits) != 0:
            proc_args += ["--target_bits", target_bits]
        if args.module:
            proc_args += ["--module", args.module]
        if args.output_dir:
            proc_args += ["--output_dir", args.output_dir]
        proc = subprocess.Popen(args=proc_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        curr_workers.append(proc)
        if len(curr_workers) >= max_workers:
            for j, process in enumerate(curr_workers):
                print("waiting for process %d" % (j + 10 * ((i - 1) / 10)))
                stdout, stderr = process.communicate()
                print(stdout.decode())
                print(stderr.decode())
            curr_workers = []
    for proc in curr_workers:
        stdout, stderr = proc.communicate()
        print(stdout.decode())
        print(stderr.decode())

        


                    
