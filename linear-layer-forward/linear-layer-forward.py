import numpy as np

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(W, list):
        W = np.array(W)
    if isinstance(X, list):
        b = np.array(b)
    
    return (X @ W + b).tolist()
