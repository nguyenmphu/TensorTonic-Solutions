import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    if rng:
        random = rng.random(x.shape)
    else:
        random = np.random.rand(x.shape)

    mask = random < (1 - p)
    dropout_pattern = mask / (1 - p)
    output = x * dropout_pattern
    return (output, dropout_pattern)
