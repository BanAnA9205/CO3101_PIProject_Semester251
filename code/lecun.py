import torch
import torch.nn as nn
import numpy as np

def calculate_fan_in(tensor):
    # tensor.ndimension() gives the number of dimensions
    if tensor.ndimension() < 2:
        raise ValueError("Fan-in cannot be computed for 1D tensor (e.g., bias).")

    n_inputs = tensor.size(1)
    receptive_field_size = 1

    # For Conv layers, multiply by kernel height × width
    # .numel(): number of elements
    if tensor.ndimension() > 2:
        receptive_field_size = tensor[0][0].numel()

    return n_inputs * receptive_field_size

def lecun_normal_init_(tensor):
    fan_in = calculate_fan_in(tensor)
    std = 1 / np.sqrt(fan_in)
    
    # note: the _ means the operation was performed inplace
    # thus, normal_ means an inplace normal distribution intialization 
    with torch.no_grad():
        return tensor.normal_(0, std)


