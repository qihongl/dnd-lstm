import torch
import numpy as np


def entropy(torch_tensor_1d):
    """calculate entropy.
    I'm using log base 2!

    Parameters
    ----------
    torch_tensor_1d : a torch vector
        a prob distribution

    Returns
    -------
    torch scalar
        the entropy of the distribution

    """
    return - torch.stack([pi * torch.log2(pi) for pi in torch_tensor_1d]).sum()


def compute_stats(vector, ddof=1):
    """compute mean and standard error

    Parameters
    ----------
    vector : 1d array
        a list of numbers
    ddof : int, optional
        Delta degrees-of-freedom
        see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html

    Returns
    -------
    float, float
        mean, standard error

    """
    n = len(vector)
    mu = np.mean(vector, axis=0)
    se = np.std(vector, axis=0) / np.sqrt(n-ddof)
    return mu, se


def to_pth(np_array, pth_dtype=torch.FloatTensor):
    '''convert numpy array -> pytorch tensor'''
    return torch.tensor(np_array).type(pth_dtype)


def to_sqpth(np_array, pth_dtype=torch.FloatTensor):
    '''convert numpy array -> pytorch tensor, then squeeze it'''
    return torch.squeeze(to_pth(np_array, pth_dtype=pth_dtype))


def to_np(torch_tensor):
    '''convert pytorch tensor -> numpy array'''
    return torch_tensor.data.numpy()


def to_sqnp(torch_tensor):
    '''convert pytorch tensor -> numpy array, then squeeze it'''
    return np.squeeze(to_np(torch_tensor))
