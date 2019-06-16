import torch


def get_init_states(dim, scale=.3):
    h_0 = torch.randn(1, 1, dim) * scale
    c_0 = torch.randn(1, 1, dim) * scale
    return h_0, c_0


def ortho_init(agent):
    for name, wts in agent.named_parameters():
        if 'weight' in name:
            torch.nn.init.orthogonal_(wts)
        elif 'bias' in name:
            torch.nn.init.constant_(wts, 0)


def softmax(z, beta):
    """helper function, softmax with beta

    Parameters
    ----------
    z : torch tensor, has 1d underlying structure after torch.squeeze
        the raw logits
    beta : float, >0
        softmax temp, big value -> more "randomness"

    Returns
    -------
    1d torch tensor
        a probability distribution | beta

    """
    assert beta > 0
    return torch.nn.functional.softmax(torch.squeeze(z / beta), dim=0)


def entropy(d_):
    """calculate entropy.
    I'm using log base 2!

    Parameters
    ----------
    d_ : a torch vector
        a prob distribution

    Returns
    -------
    torch scalar
        the entropy of the distribution

    """
    return - torch.stack([pi * torch.log2(pi) for pi in d_]).sum()
