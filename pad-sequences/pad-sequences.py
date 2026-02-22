import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    L = max_len or max(len(seq) for seq in seqs)
    mat = np.zeros((N, L))
    if pad_value:
        mat.fill(pad_value)
    for (i, seq) in enumerate(seqs):
        for j in range(min(len(seq), L)):
            mat[i, j] = seq[j]
    return mat
