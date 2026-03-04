import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    beta = np.array(beta)
    gamma = np.array(gamma)
    if x.ndim == 4:
        # Normalize over (N, H, W) per channel → move C to last, reshape to (-1, C)
        N, C, H, W = x.shape
        x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)   # (N*H*W, C)
    elif x.ndim == 2:
        x_flat = x
        C = x.shape[1]
    else:
        raise ValueError(f"Expected 2-D or 4-D input, got shape {x.shape}")

    mu     = np.mean(x_flat, axis=0)                  # (C,) — no keepdims needed
    sigma2 = np.var(x_flat, axis=0)                   # (C,) — use var(), not mean((x-mu)^2)
    x_hat  = (x_flat - mu) / np.sqrt(sigma2 + eps)   # (N*H*W, C) or (N, C)
    out    = gamma * x_hat + beta                     # broadcast over (C,)

    if x.ndim == 4:
        out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # back to (N,C,H,W)

    return out
