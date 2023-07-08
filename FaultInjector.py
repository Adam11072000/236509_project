import torch
from utils import bitwise_xor_on_floats, generate_random_indices, generate_xor_mask, build_module_dict, replace_submodule_with_faulty



#FAULT_TARGET = ["weight", "bias", "memory"] # also need number of cells to inject fault into
FAULT_TARGET = ["weight", "bias", "memory"] # also need number of cells to inject fault into
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
    def __init__(self, layer_name, fault_target="weight", num_faults=1, fault_option="strict", num_bits_to_flip=1, fault_distribution="gaussian",
                     fault_mode_per_target="random", fault_distrubution_per_target="gaussian", target_bits=()) -> None:
        if layer_name is None:
            raise WrongParameters("layer_name is not provided")
        if not fault_target or not fault_target in FAULT_TARGET:
            raise WrongParameters("fault_target parameter is wrong")
        if not fault_option in FAULT_OPTIONS or fault_option == "non-strict" and num_bits_to_flip == 1:
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
        self.done = False
    
    def __str__(self) -> str:
        representation = "layer_name-%s/%s/num-faults-%s/%s/bits-to-flip-%s/distribution-%s/target-mode-%s" % (self.layer_name, self.fault_target, 
                            self.num_faults, self.fault_option, self.num_bits_to_flip, self.fault_distribution, self.fault_mode_per_target)
        if self.fault_mode_per_target == "random":
            representation = representation + "/target-distribution-%s" % self.fault_distrubution_per_target
        else:
            representation = representation + "/target-bits-%s" % self.target_bits
        return representation


def inject_fault(target, fault_model):
    target_indices = generate_random_indices(target.shape, fault_model.num_faults, fault_model.fault_distribution)
    if target.dtype.__str__().find("float") != -1:
        func = torch.finfo
    else:
        func = torch.iinfo
    xor_mask = generate_xor_mask(func(target.dtype).bits, 0, func(target.dtype).bits - 1, fault_model.num_bits_to_flip, fault_model.fault_distrubution_per_target)
    target = bitwise_xor_on_floats(target_indices, target, xor_mask)
    return target


class FaultyModule(torch.nn.Module):
    def __init__(self, original_module, fault_model):
        super().__init__()
        self.original_module = original_module
        self.fault_model = fault_model
        self.fault_model.done = False

    def forward(self, input):
        if not self.fault_model.done:
            input = inject_fault(input.clone(), self.fault_model)
            self.fault_model.done = True
        return self.original_module(input)


def hook(module: torch.nn.Module, input: torch.tensor, output: torch.tensor, fault_model: FaultModel):
    if fault_model.done == True:
        return input
    modified_input = inject_fault(input[0].clone(), fault_model)
    input = list(input)
    input[0] = modified_input
    input = tuple(input)
    fault_model.done = True
    return modified_input


class FaultInjector():
    def __init__(self, model: torch.nn.Module, fault: FaultModel) -> None:
        self.model = model
        self.sub_modules = build_module_dict(model)
        self.fault = fault
    
    def set_submodule_fault_target(self, submodule_path, target_fault, new_tensor):
        submodule_path_list = submodule_path.split('.')
    
        # Get the submodule by recursively accessing the attributes
        submodule = self.model
        for name in submodule_path_list:
            submodule = getattr(submodule, name)
        
        # Update the weights of the submodule
        for name, param in submodule.named_parameters():
            if name == target_fault:
                param.data = new_tensor
                break
        
    def inject_faults(self):
        target = None
        target_layer = self.sub_modules[self.fault.layer_name]
        if self.fault.fault_target == "weight":
            target = target_layer.weight
        elif self.fault.fault_target == "bias":
            target = target_layer.bias
        elif self.fault.fault_target == "memory":
            faulty_module = FaultyModule(target_layer, self.fault)
            replace_submodule_with_faulty(self.model, self.fault.layer_name, faulty_module)
            return
        if target is None:
            raise Exception()
        
        target = inject_fault(target, self.fault)
        self.set_submodule_fault_target(self.fault.layer_name, self.fault.fault_target, target)
