import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from Trainer import ResNetTrainer
from utils import build_module_dict
from FaultInjector import FaultInjector, FaultModel, FAULT_TARGET, FAULT_OPTIONS, FAULT_MODE_PER_TARGET, RANDOM_DISTRIBUTIONS
import copy
from scipy.stats import truncnorm
import numpy as np
import json

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
    lower_bound_fault_num = 1
    upper_bound_fault_num = num_fault_models
    std = 2
    mean = 5
    
    # bit flips
    lower_bound_bit_flips = 2
    upper_bound_bit_flips = 7
    
    # targets
    target_choices = np.random.choice(FAULT_TARGET, size=num_fault_models)
    for i in range(0, num_fault_models):
        distribution_target = np.random.choice(RANDOM_DISTRIBUTIONS, size=1)[0]
        distribution_bit_flips = np.random.choice(RANDOM_DISTRIBUTIONS, size=1)[0]
        if distribution_bit_flips == "gaussian":
            a, b = (lower_bound_bit_flips - mean) / std, (upper_bound_bit_flips - mean) / std
            bit_flips = truncnorm.rvs(a, b, loc=mean, scale=std, size=1)
        else:
            bit_flips = np.random.uniform(low=lower_bound_bit_flips, 
                        high=upper_bound_bit_flips, size=1)
        if distribution_target == "gaussian":
            a, b = (lower_bound_fault_num - mean) / std, (upper_bound_fault_num - mean) / std
            num_faults = truncnorm.rvs(a, b, loc=mean, scale=std, size=1)
        else:
            num_faults = np.random.uniform(low=lower_bound_fault_num, # need to support gaussian
                                 high=upper_bound_fault_num, size=1)
        num_faults = int(np.around(num_faults, 0)[0])
        bit_flips = int(np.around(bit_flips, 0)[0])
        fault_option = "non-strict" # will do strict later    
        fault_target = target_choices[i]
        fault_models.append(
            FaultModel(
                layer_name=np.random.choice(supported_layers_per_fault_target[fault_target], size=1)[0]["module_name"],
                fault_target=fault_target,
                num_faults=num_faults,
                fault_option=fault_option,
                num_bits_to_flip=bit_flips,
                fault_distribution=distribution_target,
                fault_mode_per_target="random",
                fault_distrubution_per_target=distribution_bit_flips
            )
        )
    
    return fault_models

def trainSetLoader(batch_size):
    # Define transformation for the test images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, testloader


def worker(fault_model, model_path, criterion, optimizer, device, test_loader):
    model = torch.load(model_path).to(device)
    copied = copy.deepcopy(model)
    injector = FaultInjector(copied, [fault_model])
    handle = injector.inject_faults()
    trainer2 = ResNetTrainer(injector.model, criterion, optimizer, device)
    num_epochs, train_loss, train_acc, test_loss, test_acc = trainer2.fit(None, test_loader, 1)
    if handle:
        handle.remove()
    return num_epochs, train_loss, train_acc, test_loss, test_acc


if __name__ == "__main__":
    number_of_fault_models = 100
    number_of_iterations_per_fault_model = 50
    model = torchvision.models.resnet18()
    state_dict = torch.load("model_82.17.pth")
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    lr = 0.00035
    momentum = 0.95
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = ResNetTrainer(model, criterion, optimizer, device)
    _, test_loader = trainSetLoader(32)
    #trainer.fit(None, test_loader, 1)
    supported_layers_per_fault_target = get_supported_layers_per_fault_target(model)
    fault_models = generate_fault_models(supported_layers_per_fault_target, number_of_fault_models)
    overall_results = {}

    sub_modules = build_module_dict(model)
    for fault_model in fault_models:
        print(fault_model)
        appened_accuracy = []
        for i in range(0, number_of_iterations_per_fault_model):
            copied = copy.deepcopy(model)
            injector = FaultInjector(copied, fault_model)
            injector.inject_faults()
            trainer2 = ResNetTrainer(injector.model, criterion, optimizer, device)
            num_epochs, train_loss, train_acc, test_loss, test_acc = trainer2.fit(None, test_loader, 1)
            appened_accuracy.append(test_acc)
        overall_results[fault_model] = {
            "acc_per_iteration": appened_accuracy
        }
    with open("data.json", "w") as f:
        json.dump(overall_results, f)
        f.close()

                    
