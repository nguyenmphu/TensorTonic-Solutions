import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    # Position indices: (seq_len, 1)
    positions = np.arange(seq_len)[:, np.newaxis]
    
    # Dimension indices for even positions: (1, d_model/2)
    dims = np.arange(0, d_model, 2)[np.newaxis, :]
    
    # Compute the scaling factor: 1 / 10000^(2i/d_model)
    angles = positions / np.power(base, dims / d_model)
    
    # Interleave sin (even indices) and cos (odd indices)
    PE = np.zeros((seq_len, d_model))
    PE[:, 0::2] = np.sin(angles)  # even dims
    PE[:, 1::2] = np.cos(angles[:, :PE[:, 1::2].shape[1]])  # odd dims
    
    return PE
