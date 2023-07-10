import torch
import random
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_xor_mask(total_bits, min_bit, max_bit, num_bits_to_flip, target_bits=[], distribution='uniform'):
    if num_bits_to_flip > total_bits or min_bit < 0 or max_bit >= total_bits or min_bit > max_bit or \
        (len(target_bits) > 0 and (max(target_bits) >= total_bits or min(target_bits) < 0)):
        raise ValueError("Invalid input parameters")
    
    if len(target_bits) == 0:
        available_positions = list(range(min_bit, max_bit + 1))
        
        if distribution == 'uniform':
            positions_to_flip = random.sample(available_positions, num_bits_to_flip)
        elif distribution == 'gaussian':
            mean = (min_bit + max_bit) / 2.0
            stddev = (max_bit - min_bit) / 4.0
            positions_to_flip = []
            while len(positions_to_flip) < num_bits_to_flip:
                pos = int(round(random.gauss(mean, stddev)))
                if pos in available_positions and pos not in positions_to_flip:
                    positions_to_flip.append(pos)
        else:
            raise ValueError("Invalid distribution. Choose either 'gaussian' or 'uniform'.")
    else:
        positions_to_flip = target_bits
    
    xor_mask = 0
    for pos in positions_to_flip:
        xor_mask |= 1 << pos
    
    return xor_mask

def build_module_dict(model):
    module_dict = {}
    for name, module in model.named_modules():
        module_dict[name] = module
    return module_dict

def get_module_by_path(model, path):
    parts = path.split('.')
    curr_module = model
    for part in parts:
        curr_module = getattr(curr_module, part)
    return curr_module
               
def generate_random_indices(tensor_shape, cap, distribution='uniform'):
    if len(tensor_shape) < 1:
        raise ValueError("Tensor must have at least 1 dimension.")
    
    total_elements = 1
    for dim_size in tensor_shape:
        total_elements *= dim_size
    
    indices = torch.zeros(total_elements, dtype=torch.bool, device=device)
    if distribution == 'uniform':
        num_indices = min(cap, total_elements)
        rand_indices = torch.randperm(total_elements, device=device)[:num_indices]
    elif distribution == 'gaussian':
        num_indices = min(cap, total_elements)
        rand_indices = torch.round(torch.normal(mean=total_elements / 2, std=total_elements / 4, size=(num_indices,))).long()
        rand_indices = rand_indices[(rand_indices >= 0) & (rand_indices < total_elements)]
    else:
        raise ValueError("Invalid distribution. Only 'uniform' and 'gaussian' distributions are supported.")
    
    indices[rand_indices] = True
    return indices.view(*tensor_shape)


def float_to_int32(tensor):
    tensor_as_int = np.frombuffer(tensor.cpu().detach().numpy().tobytes(), dtype=np.int32)
    return torch.tensor(tensor_as_int, dtype=torch.int32).to(device)

def int32_to_float(tensor):
    tensor_as_float = np.frombuffer(tensor.cpu().detach().numpy().tobytes(), dtype=np.float32)
    return torch.tensor(tensor_as_float, dtype=torch.float32).to(device)

def bitwise_xor_on_floats(target_indices, target, xor_mask):
    tmp = target.clone()
    orig_shape = target_indices.shape
    assert target_indices.shape == tmp.shape, "Input tensors should have the same shape"

    target_int = float_to_int32(tmp)
    target_int[target_indices.flatten()] ^= xor_mask

    return int32_to_float(target_int).reshape(orig_shape)

def replace_submodule_with_faulty(module, submodule_path, faulty_module):
    # Split the submodule path into a list of submodule names
    submodule_path_list = submodule_path.split('.')
    
    # get overall paths
    parent_module = module
    target_submodule_name = ''
    for name in submodule_path_list[:-1]:
        parent_module = getattr(parent_module, name)
    target_submodule_name = submodule_path_list[-1]

    # set target module to faulty module
    setattr(parent_module, target_submodule_name, faulty_module)