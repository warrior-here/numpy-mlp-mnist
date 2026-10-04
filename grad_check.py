# @title grad_check.py
import numpy as np
def numerical_gradient(params, name, idx, X, y_one_hot, eps=1e-7):
    """
    Numerical gradient of the loss w.r.t. params[name] at position idx,
    using the centred finite difference.
    """
    original = params[name][idx]

    params[name][idx] = original + eps
    loss_plus, _ = forward(X, params)
    loss_plus = cross_entropy_loss(loss_plus, y_one_hot)

    params[name][idx] = original - eps
    loss_minus, _ = forward(X, params)
    loss_minus = cross_entropy_loss(loss_minus, y_one_hot)

    params[name][idx] = original   # restore
    return (loss_plus - loss_minus) / (2 * eps)


def gradient_check(layer_sizes, num_samples=20, eps=1e-7, seed=0):
    """
    Compare analytic gradients from backward() against numerical gradients.
    Uses a small random network and a small batch for speed.
    """
    rng = np.random.default_rng(seed)
    params = init_params(layer_sizes, seed=seed)

    # Tiny synthetic batch
    N = 8
    n_in = layer_sizes[0]
    n_out = layer_sizes[-1]
    X = rng.standard_normal((N, n_in)).astype(np.float32)
    labels = rng.integers(0, n_out, size=N)
    y_one_hot = np.eye(n_out)[labels]

    # Analytic gradients
    probs, cache = forward(X, params)
    grads = backward(cache, params, y_one_hot)

    worst = {}
    for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        param = params[name]
        gname = "d" + name
        analytic = grads[gname]

        # Pick num_samples random indices in the parameter
        flat_size = param.size
        picks = rng.choice(flat_size, size=min(num_samples, flat_size), replace=False)

        rel_errors = []
        for p in picks:
            idx = np.unravel_index(p, param.shape)
            num_grad = numerical_gradient(params, name, idx, X, y_one_hot, eps)
            ana_grad = analytic[idx]
            denom = max(abs(num_grad), abs(ana_grad), 1e-12)
            rel_errors.append(abs(num_grad - ana_grad) / denom)

        worst[name] = max(rel_errors)
        verdict = "OK" if worst[name] < 1e-5 else "CHECK"
        print(f"{name}: worst relative error = {worst[name]:.2e}  [{verdict}]")

    return worst


# Run it on a small net
worst = gradient_check([20, 10, 8, 10], num_samples=20)