import torch
import torchvision
from Trainer import ResNetTrainer
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import copy
from FaultInjector import FaultInjector, FaultModel, WrongParameters
import argparse
import json
import os


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="This script initializes a FaultModel object with given parameters.")

parser.add_argument("-l", "--layer_name", required=True, help="The name of the layer.")
parser.add_argument("-f", "--fault_target", default="weight", help="The target of the fault. Default is 'weight'.")
parser.add_argument("-n", "--num_faults", type=int, default=1, help="The number of faults. Default is 1.")
parser.add_argument("-o", "--fault_option", default="non-strict", help="The fault option. Default is 'non-strict'.")
parser.add_argument("-b", "--num_bits_to_flip", type=int, default=1, help="The number of bits to flip. Default is 1.")
parser.add_argument("-d", "--fault_distribution", default="gaussian", help="The fault distribution. Default is 'gaussian'.")
parser.add_argument("-m", "--fault_mode_per_target", default="random", help="The fault mode per target. Default is 'random'.")
parser.add_argument("-p", "--fault_distrubution_per_target", default="gaussian", help="The fault distribution per target. Default is 'gaussian'.")
parser.add_argument("-t", "--target_bits", nargs='*', default=(), help="The target bits. Provide multiple values separated by space. Default is an empty list.")
parser.add_argument('-i', '--number_of_iterations_per_fault_model', type=int, default=1,
                    help='Number of iterations per fault (must be higher than 0)')
parser.add_argument('-id', '--worker_id', type=int, required=True,
                    help='ID of worker')

args = parser.parse_args()

WORKERS_DIR = "./workers_data"
TARGET_JSON = "%s/worker_data_%d.json" % (WORKERS_DIR, args.worker_id)

if not os.path.isdir(WORKERS_DIR):
    os.makedirs(WORKERS_DIR)


if len(args.target_bits) > 0:
    args.target_bits = [int(bit) for bit in args.target_bits]

try:
    fault_model = FaultModel(layer_name=args.layer_name, 
                        fault_target=args.fault_target, 
                        num_faults=args.num_faults, 
                        fault_option=args.fault_option, 
                        num_bits_to_flip=args.num_bits_to_flip, 
                        fault_distribution=args.fault_distribution,
                        fault_mode_per_target=args.fault_mode_per_target,
                        fault_distrubution_per_target=args.fault_distrubution_per_target,
                        target_bits=args.target_bits)
except WrongParameters as e:
    print(e)


device = "cuda" if torch.cuda.is_available() else "cpu"

def trainSetLoader(batch_size):
    # Define transformation for the test images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    return testloader

model = torchvision.models.resnet18()
state_dict = torch.load("model_82.17.pth")
model.load_state_dict(state_dict)
model.eval()

criterion = nn.CrossEntropyLoss()
lr = 0.00035
momentum = 0.95
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
trainer = ResNetTrainer(model, criterion, optimizer, device)
test_loader = trainSetLoader(32)

appened_accuracy = []
overall_results = {}
overall_results["fault_mode_%d" % args.worker_id] = copy.deepcopy(fault_model.__dict__())
for i in range(0, args.number_of_iterations_per_fault_model):
    injector = FaultInjector(model, fault_model)
    injector.inject_faults()
    trainer2 = ResNetTrainer(injector.model, criterion, optimizer, device)
    num_epochs, train_loss, train_acc, test_loss, test_acc = trainer2.fit(None, test_loader, 1)
    appened_accuracy.append(test_acc[0])

overall_results["fault_mode_%d" % args.worker_id]["acc_per_iteration"] = appened_accuracy
with open(TARGET_JSON, "w") as f:
    json.dump(overall_results, f)
    f.close()

exit(0)