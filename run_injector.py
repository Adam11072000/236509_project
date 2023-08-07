import torchvision
import torch
from utils import build_module_dict
from FaultInjector import FaultModel, FAULT_TARGET, RANDOM_DISTRIBUTIONS, WrongParameters
import numpy as np
import subprocess
import argparse
import robustbench
import os

# Add the arguments
parser = argparse.ArgumentParser(description='Arguments for the script. For the list of available models, check https://github.com/RobustBench/robustbench')

parser.add_argument('-n', '--number_of_fault_models', type=int, default=1,
                    help='Number of fault models (must be higher than 0)')
parser.add_argument('-i', '--number_of_iterations_per_fault', type=int, default=1,
                    help='Number of iterations per fault (must be higher than 0)')
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
parser.add_argument("--interactive", action='store_true', help="Interactive fault model choosing")
parser.add_argument("-d", "--distribution_faults", type=str, help="Distribution of faults", choices=RANDOM_DISTRIBUTIONS)
parser.add_argument("--force", action="store_true", help="force run regardless if output dir is present")

# Parse the arguments
args = parser.parse_args()

# Ensure number_of_faults, number_of_iterations_per_fault, and num_bits_to_flip are greater than 0
if args.number_of_fault_models < 1:
    parser.error("number_of_faults must be higher than 0")

if args.number_of_iterations_per_fault < 1:
    parser.error("number_of_iterations_per_fault must be higher than 0")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

if not args.force and args.output_dir and os.path.isdir(args.output_dir):
    print("Skipping because output dir is present and non-forced run is used")
    exit(0)

print(f"Using {device}")


def pretty_print_supported_layers(layers_dict):
    for main_key in layers_dict:
        print(f"{main_key}:")
        for sub_dict in layers_dict[main_key]:
            print(f"\t{sub_dict['module_name']}")

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
    
    # bit flips
    lower_bound_bit_flips = 1
    upper_bound_bit_flips = 7
    
    # targets
    target_choices = np.random.choice(FAULT_TARGET, size=num_fault_models)
    for i in range(0, num_fault_models):
        # random choosers 
        distribution_target = np.random.choice(RANDOM_DISTRIBUTIONS, size=1)[0]
        if args.distribution_faults:
            distribution_target = args.distribution_faults
        distribution_bit_flips = np.random.choice(RANDOM_DISTRIBUTIONS, size=1)[0]

        target_bits = []
        if args.target_bits:
            bit_flips = len(args.target_bits)
            target_bits = args.target_bits
        else:
            bit_flips = int(np.around(np.random.uniform(low=lower_bound_bit_flips, 
                        high=upper_bound_bit_flips, size=1), 0)[0])
        target_bits = [int(bit) for bit in target_bits]   
        
        if args.num_faults:
            num_faults = args.num_faults
        else:
            num_faults = int(np.around(np.random.uniform(low=lower_bound_fault_num,
                                 high=upper_bound_fault_num, size=1), 0)[0])

        #fault target decisions
        fault_target = target_choices[i]
        if args.fault_target:
            fault_target = args.fault_target

        # layer name decisions
        layer_name = np.random.choice(supported_layers_per_fault_target[fault_target], size=1)[0]["module_name"]
        if args.layer_name and not args.interactive:
            layer_name = args.layer_name
            if not any(layer["module_name"] == layer_name for layer in supported_layers_per_fault_target[fault_target]):
                raise WrongParameters("layer_name is not supported with given target")
        if args.interactive:
            module_names = [sub_dict['module_name'] for sub_dict in supported_layers_per_fault_target[fault_target]]
            print("Please enter a layer name from below:")
            print(f"{fault_target}: {' '.join(module_names)}")
            layer_name = input()
        
        # creation of fault
        fault_models.append(
            FaultModel(
                layer_name=layer_name,
                fault_target=fault_target,
                num_faults=num_faults,
                num_bits_to_flip=bit_flips,
                fault_distribution=distribution_target,
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
        state_dict = torch.load("model_82.17.pth", map_location=device)
        model.load_state_dict(state_dict)
        # model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.DEFAULT)
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
        fault_distrubution_per_target = fault_model.fault_distrubution_per_target
        target_bits = ' '.join(str(bit) for bit in fault_model.target_bits)
        worker_id = i
        proc_args = [
            "python3", "worker.py", "--layer_name", layer_name, "--fault_target", fault_target,
            "--num_faults", str(num_faults), "--num_bits_to_flip", str(num_bits_to_flip),
            "--fault_distribution", fault_distribution,
            "--fault_distrubution_per_target", fault_distrubution_per_target,
            "--number_of_iterations_per_fault_model", str(number_of_iterations_per_fault_model), "--worker_id", str(worker_id)
        ]
        if len(fault_model.target_bits) != 0:
            proc_args += ["--target_bits", target_bits]
        if args.module:
            proc_args += ["--module", args.module]
        if args.output_dir:
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
            proc_args += ["--output_dir", args.output_dir]
        print(' '.join(proc_args))
        proc = subprocess.Popen(args=proc_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        curr_workers.append(proc)
        if len(curr_workers) >= max_workers:
            for j, process in enumerate(curr_workers):
                print("waiting for process %d" % (j + 10 * (i / 10)))
                stdout, stderr = process.communicate()
                print(stdout.decode())
                print(stderr.decode())
                proc.stdout.close()
                proc.stderr.close()
            curr_workers = []
    for proc in curr_workers:
        stdout, stderr = proc.communicate()
        print(stdout.decode())
        print(stderr.decode())
        proc.stdout.close()
        proc.stderr.close()

        


                    
