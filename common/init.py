import torch.nn as nn
from torch.nn.init import constant_, kaiming_uniform_, xavier_normal_, xavier_uniform_


def xavier_normal_initialization(module):
    r"""using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight)
    elif isinstance(module, nn.Parameter):
        if module.dim() == 1:
            xavier_uniform_(module.unsqueeze(0)).squeeze(0)
        else:
            xavier_uniform_(module)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight)
        if module.bias is not None:
            constant_(module.bias, 0)
    # recursively handle sub-modules
    elif isinstance(module, nn.ModuleDict):
        for sub_module in module.values():
            xavier_normal_initialization(sub_module)
    elif isinstance(module, nn.ModuleList):
        for sub_module in module:
            xavier_normal_initialization(sub_module)
    elif isinstance(module, nn.ParameterDict):
        for sub_module in module.values():
            xavier_normal_initialization(sub_module)


def xavier_uniform_initialization(module):
    r"""using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

    """
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight)
    elif isinstance(module, nn.Parameter):
        if module.dim() == 1:
            xavier_uniform_(module.unsqueeze(0)).squeeze(0)
        else:
            xavier_uniform_(module)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight)
        if module.bias is not None:
            constant_(module.bias, 0)
    # recursively handle sub-modules
    elif isinstance(module, nn.ModuleDict):
        for sub_module in module.values():
            xavier_uniform_initialization(sub_module)
    elif isinstance(module, nn.ModuleList):
        for sub_module in module:
            xavier_uniform_initialization(sub_module)
    elif isinstance(module, nn.ParameterDict):
        for sub_module in module.values():
            xavier_uniform_initialization(sub_module)


def kaiming_uniform_initialization(module, nonlinearity="leaky_relu", a=0.2):
    r"""using `kaiming_uniform_`_ in PyTorch to initialize the parameters in
    nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`kaiming_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=kaiming_uniform_#torch.nn.init.kaiming_uniform_

    """
    if isinstance(module, nn.Linear):
        kaiming_uniform_(module.weight, a=a, nonlinearity=nonlinearity)
        if module.bias is not None:
            constant_(module.bias, 0)
