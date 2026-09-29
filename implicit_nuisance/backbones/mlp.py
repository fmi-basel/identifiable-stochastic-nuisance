import torch
import torch.nn as nn

def MLP(input_dim, hidden_dim, output_dim, n_hidden_layers=1, activation=nn.ReLU, output_activation=nn.Identity, bias=True):
    """
    Create a Multi-Layer Perceptron (MLP) model.
    
    Parameters:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units in each layer.
        output_dim (int): Number of output features.
        n_hidden_layers (int): Number of hidden layers.
        activation (callable): Activation function to use for hidden layers.
        output_activation (callable): Activation function to use for output layer.
        bias (bool): Whether to use bias in the linear layers.
    
    Returns:
        torch.nn.Sequential: The MLP model.
    """
    layers = []
    if n_hidden_layers == 0:
        layers.append(nn.Linear(input_dim, output_dim, bias=bias))
    else:
        layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
        layers.append(activation())
        
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            layers.append(activation())
        
        layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    layers.append(output_activation())
    
    return nn.Sequential(*layers)
