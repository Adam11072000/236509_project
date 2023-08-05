import torch
from utils import bitwise_xor_on_floats, generate_random_indices, generate_xor_mask, build_module_dict, replace_submodule_with_faulty



FAULT_TARGET = ["weight", "bias", "memory"] # also need number of cells to inject fault into
RANDOM_DISTRIBUTIONS = ["gaussian", "uniform"]  #  geometric

class WrongParameters(Exception):
    def __init__(self, msg, *args: object) -> None:
        self.msg = msg
        super().__init__(*args)
    def __str__(self) -> str:
        return self.msg


class FaultModel():
    def __init__(self, layer_name, fault_target="weight", num_faults=1, num_bits_to_flip=1, fault_distribution="gaussian", fault_distrubution_per_target="gaussian", target_bits=[]) -> None:
        if layer_name is None:
            raise WrongParameters("layer_name is not provided")
        if not fault_target or not fault_target in FAULT_TARGET:
            raise WrongParameters("fault_target parameter is wrong")
        self.fault_target = fault_target
        self.num_faults = num_faults
        self.num_bits_to_flip = num_bits_to_flip
        self.fault_distribution = fault_distribution
        self.fault_distrubution_per_target = fault_distrubution_per_target
        self.target_bits = target_bits
        self.layer_name = layer_name
        self.done = False
    
    def __str__(self) -> str:
        representation = "layer-name_%s/%s/num-faults_%s/bits-to-flip_%s/distribution_%s" % (self.layer_name, self.fault_target, 
                            self.num_faults, self.num_bits_to_flip, self.fault_distribution)
        if len(self.target_bits) == 0:
            representation = representation + "/target-distribution_%s" % self.fault_distrubution_per_target
        else:
            representation = representation + "/target-bits_%s" % '-'.join([str(i) for i in self.target_bits])
        return representation

    def __dict__(self) -> dict:
        return {
            "fault_target": self.fault_target,
            "num_faults": self.num_faults,
            "num_bits_to_flip": self.num_bits_to_flip,
            "fault_distribution": self.fault_distribution,
            "fault_distrubution_per_target": self.fault_distrubution_per_target,
            "target_bits": self.target_bits,
            "layer_name": self.layer_name
        }


def inject_fault(target, fault_model):
    target_indices = generate_random_indices(target.shape, fault_model.num_faults, fault_model.fault_distribution)
    if target.dtype.__str__().find("float") != -1:
        func = torch.finfo
    else:
        func = torch.iinfo
    xor_mask = generate_xor_mask(func(target.dtype).bits, 0, func(target.dtype).bits - 1, 
                                    fault_model.num_bits_to_flip, fault_model.target_bits, fault_model.fault_distrubution_per_target)
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


class FaultInjector():
    def __init__(self, model: torch.nn.Module, fault: FaultModel) -> None:
        self.model = model
        self.sub_modules = build_module_dict(model)
        self.fault = fault
    
    def set_submodule_fault_target(self, submodule_path, target_fault, new_tensor):
        submodule_path_list = submodule_path.split('.')
        submodule = self.model
        for name in submodule_path_list:
            submodule = getattr(submodule, name)
        
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
            raise Exception("target is None, something is wrong")
        
        target = inject_fault(target, self.fault)
        self.set_submodule_fault_target(self.fault.layer_name, self.fault.fault_target, target)
