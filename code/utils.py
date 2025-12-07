import torch
import torch.nn as nn
import numpy as np


def calculate_fan_in(tensor):
    """
    Calculate the fan-in (number of input units) for a given weight tensor.
    Examples:
    - For a Linear layer with shape (out_features, in_features), 
        fan-in = in
    - For a Conv2d layer with shape (out_channels, in_channels, kernel_height, kernel_width), 
        fan-in = in_channels * kernel_height * kernel_width
    """
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
    """
    The LeCun normal initialization initializes the weights from a normal distribution
    with mean 0 and standard deviation sqrt(1 / fan_in)
    where fan_in is the number of input units in the weight tensor.

    This initialization is particularly well-suited for activation functions like
    the Linear and Sigmoid/Tanh functions since it preserves the variance of
    activations throughout the network layers. 

    (see LeCun et al., 1998)
    """
    fan_in = calculate_fan_in(tensor)
    std = np.sqrt(1 / fan_in)
    
    # note: the _ means the operation was performed inplace
    # thus, normal_ means an inplace normal distribution intialization 
    with torch.no_grad():
        return tensor.normal_(0, std)

def kaiming_normal_init_(tensor):
    """
    The Kaiming normal initialization (also known as He initialization) initializes
    the weights from a normal distribution with mean 0 and standard deviation sqrt(2 / fan_in)
    where fan_in is the number of input units in the weight tensor.

    This initialization is particularly well-suited for ReLU activation functions
    since it helps maintain the variance of activations throughout the network layers.

    (see He et al., 2015)
    """
    fan_in = calculate_fan_in(tensor)
    std = np.sqrt(2 / fan_in)
    
    with torch.no_grad():
        return tensor.normal_(0, std)