import torch
import random
import struct


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


def generate_xor_mask(total_bits, min_bit, max_bit, num_bits_to_flip, distribution='uniform'):
    # Ensure that num_bits_to_flip, min_bit, and max_bit are valid
    if num_bits_to_flip > total_bits or min_bit < 0 or max_bit >= total_bits or min_bit > max_bit:
        raise ValueError("Invalid input parameters")
    
    # Adjust the range of positions to be between min_bit and max_bit
    available_positions = list(range(min_bit, max_bit + 1))
    
    # Randomly select num_bits_to_flip unique positions within the range
    if distribution == 'uniform':
        positions_to_flip = random.sample(available_positions, num_bits_to_flip)
    elif distribution == 'gaussian':
        # Mean and standard deviation for gaussian distribution
        mean = (min_bit + max_bit) / 2.0
        stddev = (max_bit - min_bit) / 4.0
        positions_to_flip = []
        while len(positions_to_flip) < num_bits_to_flip:
            pos = int(round(random.gauss(mean, stddev)))
            if pos in available_positions and pos not in positions_to_flip:
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
    
    # Initialize the boolean tensor with all elements set to False
    indices = torch.zeros(tensor_shape, dtype=torch.bool)
    
    # Flatten the last two dimensions for index calculation
    flat_size = tensor_shape[-2] * tensor_shape[-1]

    # Generate random indices based on the specified distribution
    if distribution == 'uniform':
        num_indices = min(cap, flat_size)
        rand_indices = torch.randperm(flat_size)[:num_indices]
    elif distribution == 'gaussian':
        num_indices = min(cap, flat_size)
        while True:
            rand_indices = torch.normal(mean=flat_size / 2, std=flat_size / 4, size=(num_indices,)).long()
            rand_indices = rand_indices[rand_indices >= 0]
            rand_indices = rand_indices[rand_indices < flat_size]
            if len(rand_indices) >= num_indices:
                rand_indices = rand_indices[:num_indices]
                break
    else:
        raise ValueError("Invalid distribution. Only 'uniform' and 'gaussian' distributions are supported.")
    
    for index in rand_indices:
        # Randomly select indices for the first dimensions
        rand_first_dims = tuple(random.randrange(0, dim) for dim in tensor_shape[:-2])
        
        # Convert linear index to subscript for the last two dimensions
        row = index // tensor_shape[-1]
        col = index % tensor_shape[-1]
        
        # Set the index to True
        indices[rand_first_dims + (row, col)] = True

    return indices

def float_to_int_repr(f):
    """Converts a float number to its 32-bit integer representation"""
    return struct.unpack('!I', struct.pack('!f', f))[0]

def int_repr_to_float(i):
    """Converts a 32-bit integer representation to a float number"""
    return struct.unpack('!f', struct.pack('!I', i))[0]

def flip_bit(number, bit_position):
    """Flips a particular bit of an integer"""
    return number ^ (1 << bit_position)

def bitwise_xor_on_floats(target_indices, target, xor_mask):
    # Ensure the input tensors have the same shape    
    orig_shape = target_indices.shape

    target_indices = torch.flatten(target_indices).detach()
    target = torch.flatten(target).detach()
    assert len(target.shape) == 1, "fuck"
    target.apply_(float_to_int_repr)
    result = []
    for xor, el in zip(target_indices, target):
        if not xor:
            print(el.dtype)
            result.append(el)
        else:
            result.append(el ^ xor_mask)
    result = torch.tensor(result).reshape(orig_shape).apply_(int_repr_to_float)
    return result