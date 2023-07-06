import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from Trainer import ResNetTrainer
from utils import flatten_dict_into_list, parse_submodules
from FaultInjector import FaultInjector, FaultModel

device = "cuda" if torch.cuda.is_available() else "cpu"



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

import copy

if __name__ == "__main__":
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    lr = 0.00035
    momentum = 0.95
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = ResNetTrainer(model, criterion, optimizer, device)
    train_loader, test_loader = trainSetLoader(32)
    trainer.fit(train_loader, test_loader, 10)
    sub_modules = flatten_dict_into_list(parse_submodules(model))
    injector = FaultInjector(model, [FaultModel(
        layer_name=sub_modules[3],
        fault_target="weight",
        num_faults=0,
        fault_option="strict",
        num_bits_to_flip=1,
        target_bits=(30)
    )])
    injector.inject_faults()
    trainer2 = ResNetTrainer(injector.model, criterion, optimizer, device)
    trainer2.fit(None, test_loader, 1)

