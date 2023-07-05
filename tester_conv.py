import torch
import random
import math
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from time import time
from torch.utils.data import DataLoader
import tqdm
import abc
import os
from typing import List, NamedTuple
import sys
from typing import Any, Callable
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """

    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """

    losses: List[float]
    accuracy: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """

    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]



class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs,
        print_every=1,
        model_type="Regular",
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None

        models = []
        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            # TODO:
            #  Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            #  - Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======
            
            start_time = time()
            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            epoch_train_time = time() - start_time
            train_loss += train_result.losses
            train_acc += [train_result.accuracy]
            start_time = time()
            test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            epoch_test_time = time() - start_time
            test_loss += test_result.losses
            test_acc += [test_result.accuracy]

            if best_acc is None or test_result.accuracy > best_acc:
                best_acc = test_result.accuracy

            # ========================
            file_name = model_type + "_%d" % epoch + ".pt"
            model_name = "model_%s_%d_dic.json" % (model_type, epoch)
            curr_model = {
                "model_name": model_name,
                "train_accuracy": train_result.accuracy,
                "test_accuracy": test_result.accuracy,
                "epoch_train_time": epoch_train_time,
                "epoch_test_time": epoch_test_time
            }
            models.append(curr_model)
            torch.save(self.model.state_dict(), file_name)
            with open(model_name, "w") as f:
                json.dump(curr_model, f)
                f.close()

        return FitResult(num_epochs, train_loss, train_acc, test_loss, test_acc), models

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        return EpochResult(losses=losses, accuracy=accuracy)


class ResNetTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        super().__init__(model, loss_fn, optimizer, device)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        out = self.model(inputs)
        loss = self.loss_fn(out, labels)
        loss.backward()
        self.optimizer.step()
        _, preds = torch.max(out.data, dim=1)
        num_correct = torch.sum(preds == labels)
        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        with torch.no_grad():
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            out = self.model(inputs)
            loss = self.loss_fn(out, labels)
            _, preds = torch.max(out.data, dim=1)
            num_correct = torch.sum(preds == labels)
            return BatchResult(loss.item(), num_correct.item())

FAULT_TARGET = ["weights", "bias", "memory"] # also need number of cells to inject fault into
FAULT_OPTIONS = ["strict", "non-strict"] # strict indicates 1 bit, non-strict indicates multiple bit flips
FAULT_MODE_PER_TARGET = ["random", "deterministic"] # fault type on data, either random faults across the martrices or deterministic faults
RANDOM_DISTRIBUTIONS = ["gaussian", "uniform"]  #  geometric

class WrongParameters(Exception):
    def __init__(self, msg, *args: object) -> None:
        self.msg = msg
        super().__init__(*args)
    def __str__(self) -> str:
        return self.msg


class FaultModel():
    def __init__(self, layer_name, fault_target="weights", num_faults=1, fault_option="strict", num_bits_to_flip=1, fault_distribution="gaussian",
                     fault_mode_per_target="random", fault_distrubution_per_target="gaussian", target_bits=()) -> None:
        if not layer_name:
            raise WrongParameters("layer_name is not provided")
        if not fault_target or not fault_target in FAULT_TARGET:
            raise WrongParameters("fault_target parameter is wrong")
        if not fault_option in FAULT_OPTIONS or fault_option == "non-strict" and num_bits_to_flip != 1:
            raise WrongParameters("fault_option parameter or num_faults are wrong")
        if not fault_mode_per_target in FAULT_MODE_PER_TARGET or fault_mode_per_target == "deterministic" and len(target_bits) == 0:
            raise WrongParameters("fault_mode_per_target parameter or target_bits are wrong")
        self.fault_target = fault_target
        self.num_faults = num_faults
        self.fault_option = fault_option
        self.num_bits_to_flip = num_bits_to_flip
        self.fault_distribution = fault_distribution
        self.fault_distrubution_per_target = fault_distrubution_per_target
        self.target_bits = target_bits
        self.fault_mode_per_target = fault_mode_per_target
        self.layer_name = layer_name
    
    def __str__(self) -> str:
        representation = "%s/num-faults-%s/%s/bits-to-flip-%s/distribution-%s/target-mode-%s" % (self.fault_target, 
                            self.num_faults, self.fault_option, self.num_bits_to_flip, self.fault_distribution, self.fault_mode_per_target)
        if self.fault_mode_per_target == "random":
            representation = representation + "/target-distribution-%s" % self.fault_distrubution_per_target
        else:
            representation = representation + "/target-bits-%s" % self.target_bits
        return representation

def flatten_dict_into_list(nested_dict):
    flattened_dict = []
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            flattened_dict += flatten_dict_into_list(value)
        else:
            flattened_dict.append(key)
    return flattened_dict

def parse_submodules(module, prefix='', parent_is_subscriptable=True):
    result = {}
    for idx, (name, submodule) in enumerate(module.named_children()):
        # Check if the submodule is subscriptable (e.g. nn.Sequential)
        is_subscriptable = hasattr(submodule, '__getitem__')
        
        # Construct the access path
        if parent_is_subscriptable:
            new_prefix = f"{prefix}[{idx}]" if prefix else f"{name}"
            access_path = f"{new_prefix}"
        else:
            new_prefix = f"{prefix}.{name}" if prefix else name
            access_path = f"{new_prefix}"
        
        # If the submodule has children, recursively process them
        children = dict(submodule.named_children())
        if children:
            result[access_path] = parse_submodules(submodule, new_prefix, is_subscriptable)
        else:
            result[access_path] = None
    return result

def generate_xor_mask(x, y, distribution='uniform'):
    # Ensure that y is not greater than x
    if y > math.log2(x):
        raise ValueError("Number of bits to flip should be less than or equal to the total bits")
    
    # Randomly select y unique positions to flip
    if distribution == 'uniform':
        positions_to_flip = random.sample(range(x), y)
    elif distribution == 'gaussian':
        # Mean and standard deviation for gaussian distribution
        mean = x / 2.0
        stddev = x / 4.0
        positions_to_flip = []
        while len(positions_to_flip) < y:
            pos = int(round(random.gauss(mean, stddev)))
            if 0 <= pos < x and pos not in positions_to_flip:
                positions_to_flip.append(pos)
    else:
        raise ValueError("Invalid distribution. Choose either 'gaussian' or 'uniform'.")
    
    # Build the mask by setting the selected positions to 1
    xor_mask = 0
    for pos in positions_to_flip:
        xor_mask |= 1 << pos
    
    return xor_mask

               
def generate_random_indices(tensor_shape, cap, distribution='uniform'):
    if len(tensor_shape) < 2:
        raise ValueError("Tensor must have at least 2 dimensions.")

    indices_shape = tensor_shape[:-2] + (tensor_shape[-2], tensor_shape[-1])
    indices = torch.zeros(indices_shape, dtype=torch.uint8)

    # Generate random indices based on the specified distribution
    if distribution == 'uniform':
        num_indices = min(cap, tensor_shape[-2] * tensor_shape[-1])
        rand_indices = torch.randperm(tensor_shape[-2] * tensor_shape[-1])[:num_indices]
    elif distribution == 'gaussian':
        num_indices = min(cap, tensor_shape[-2] * tensor_shape[-1])
        rand_indices = torch.clamp(torch.round(torch.randn(num_indices) * (min(cap, tensor_shape[-2] * tensor_shape[-1]) / 2) + (tensor_shape[-2] * tensor_shape[-1] / 2)), 0, tensor_shape[-2] * tensor_shape[-1] - 1)
    else:
        raise ValueError("Invalid distribution. Only 'uniform' and 'gaussian' distributions are supported.")

    # Set the selected indices to True
    indices.view(-1)[rand_indices] = 1

    return indices


class FaultInjector():
    def __init__(self, model: torch.nn.Module, faults: list[FaultModel]) -> None:
        self.model = model
        self.faults = faults
        self.sub_modules = parse_submodules(model)
    
    def inject_faults(self):
        for fault in self.faults:
            target_layer = getattr(self.model, fault.layer_name)
            target = target_layer.weights if fault.fault_option == "weights" and target_layer.weights \
                else target_layer.bias if fault.fault_option == "bias" and target_layer.bias\
                else target_layer.input if fault.fault_option == "memory" and target_layer.input\
                else None
            if not target:
                raise Exception()
            target_indices = generate_random_indices(target.shape, fault.num_faults, fault.fault_distribution)
            xor_mask = generate_xor_mask(target.dtype.itemsize, fault.num_bits_to_flip, fault.fault_distrubution_per_target)
            result_tensor = torch.bitwise_xor(target_indices * xor_mask, target)
            target = result_tensor


def trainSetLoader(batch_size):
    # Define transformation for the test images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader, testloader

if __name__ == "__main__":
    resnet18 = models.resnet18()
    resnet18.to(device)
    resnet18.train()
    criterion = nn.CrossEntropyLoss()
    lr = 0.0003
    momentum = 0.95
    optimizer = optim.SGD(resnet18.parameters(), lr=lr, momentum=momentum)
    trainer = ResNetTrainer(resnet18, criterion, optimizer)
    train_loader, test_loader = trainSetLoader(512)
    trainer.fit(train_loader, test_loader, 8)

