"""
model.py — NumPy-only MLP for MNIST classification.

Network: 784 -> 128 -> 64 -> 10, ReLU hidden, softmax output, cross-entropy loss.
Steps covered: initialisation, forward pass, loss, backprop.
"""

import numpy as np


# ------------------------------------------------------------------
# Step 2 — Initialisation (He initialisation for ReLU networks)
# ------------------------------------------------------------------
def init_params(layer_sizes, seed=42):
    """
    He initialisation for a fully connected network.
    layer_sizes: list of layer widths, e.g. [784, 128, 64, 10].
    Returns a dict {W1, b1, W2, b2, ...}.
    """
    rng = np.random.default_rng(seed)
    params = {}
    for i in range(1, len(layer_sizes)):
        n_in  = layer_sizes[i - 1]
        n_out = layer_sizes[i]
        params[f"W{i}"] = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        params[f"b{i}"] = np.zeros(n_out)
    return params


# ------------------------------------------------------------------
# Step 3 — Forward pass (linear, ReLU, softmax)
# ------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)


def softmax(z):
    # Subtract row-wise max for numerical stability.
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def forward(X, params):
    """
    Forward pass through 784 -> 128 -> 64 -> 10.
    Returns:
        probs: (batch, 10) softmax probabilities
        cache: dict of intermediates needed by backprop
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    W3, b3 = params["W3"], params["b3"]

    Z1 = X  @ W1 + b1          # (batch, 128)
    A1 = relu(Z1)              # (batch, 128)

    Z2 = A1 @ W2 + b2          # (batch, 64)
    A2 = relu(Z2)              # (batch, 64)

    Z3 = A2 @ W3 + b3          # (batch, 10)
    probs = softmax(Z3)        # (batch, 10)

    cache = {
        "X": X,
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "probs": probs,
    }
    return probs, cache


# ------------------------------------------------------------------
# Step 4 — Cross-entropy loss (numerically stable)
# ------------------------------------------------------------------
def cross_entropy_loss(probs, y_one_hot):
    """
    Average cross-entropy loss over a batch.
    probs:     (batch, 10) softmax outputs
    y_one_hot: (batch, 10) one-hot labels
    """
    N = probs.shape[0]
    probs_clipped = np.clip(probs, 1e-12, 1.0)
    log_probs = np.log(probs_clipped)
    loss = -np.sum(y_one_hot * log_probs) / N
    return loss


# ------------------------------------------------------------------
# Step 5 — Backpropagation
# ------------------------------------------------------------------
def backward(cache, params, y_one_hot):
    """
    Backprop through 784 -> 128 -> 64 -> 10 with softmax + cross-entropy.
    Returns a dict of gradients {dW1, db1, dW2, db2, dW3, db3}.
    """
    X      = cache["X"]
    Z1, A1 = cache["Z1"], cache["A1"]
    Z2, A2 = cache["Z2"], cache["A2"]
    probs  = cache["probs"]

    W2, W3 = params["W2"], params["W3"]
    N = X.shape[0]

    # Output layer: softmax + cross-entropy combine to (probs - y)/N.
    dZ3 = (probs - y_one_hot) / N              # (N, 10)
    dW3 = A2.T @ dZ3                           # (64, 10)
    db3 = dZ3.sum(axis=0)                      # (10,)
    dA2 = dZ3 @ W3.T                           # (N, 64)

    # Hidden layer 2.
    dZ2 = dA2 * (Z2 > 0)                       # ReLU derivative
    dW2 = A1.T @ dZ2                           # (128, 64)
    db2 = dZ2.sum(axis=0)                      # (64,)
    dA1 = dZ2 @ W2.T                           # (N, 128)

    # Hidden layer 1.
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1                            # (784, 128)
    db1 = dZ1.sum(axis=0)                      # (128,)

    return {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2,
        "dW3": dW3, "db3": db3,
    }


# ------------------------------------------------------------------
# Convenience: single-call forward + loss + backward
# ------------------------------------------------------------------
def forward_backward(X, y_one_hot, params):
    """
    One step of forward + loss + backward. Handy inside the training loop.
    """
    probs, cache = forward(X, params)
    loss = cross_entropy_loss(probs, y_one_hot)
    grads = backward(cache, params, y_one_hot)
    return loss, grads, probs


# ------------------------------------------------------------------
# Self-test: run this file alone to sanity-check all four steps.
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 8

    # Fake MNIST-shaped batch
    X = rng.standard_normal((N, 784)).astype(np.float32)
    labels = rng.integers(0, 10, size=N)
    y_one_hot = np.eye(10, dtype=np.float32)[labels]

    params = init_params([784, 128, 64, 10])

    # Init check
    print("Init:")
    for name, value in params.items():
        print(f"  {name}: shape={value.shape}, std={value.std():.4f}")

    # Forward
    probs, cache = forward(X, params)
    print(f"\nForward: probs.shape={probs.shape}, row_sums={probs.sum(axis=1).round(6)}")

    # Loss (should be near log(10) ~ 2.3026 at init)
    loss = cross_entropy_loss(probs, y_one_hot)
    print(f"Loss at init: {loss:.4f}  (expect ~{np.log(10):.4f})")

    # Backward
    grads = backward(cache, params, y_one_hot)
    print("\nBackward: gradient shapes vs parameter shapes")
    for g in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        p = g[1:]
        match = "OK" if grads[g].shape == params[p].shape else "MISMATCH"
        print(f"  {g}: {grads[g].shape} vs {p} {params[p].shape}  [{match}]")