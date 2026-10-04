"""
train.py — load MNIST, train the MLP with mini-batch SGD + step LR decay,
evaluate on the test set, and save learning curves + confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt

from data_loader import (
    X_train, y_train, y_train_one_hot,
    X_val, y_val,
    X_test, y_test,
)
from model import init_params, forward, cross_entropy_loss, backward


# ------------------------------------------------------------------
# Prediction and parameter update helpers
# ------------------------------------------------------------------
def predict(X, params):
    probs, _ = forward(X, params)
    return probs.argmax(axis=1)


def accuracy(X, y, params):
    return (predict(X, params) == y).mean()


def sgd_update(params, grads, lr):
    for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        params[name] -= lr * grads["d" + name]


# ------------------------------------------------------------------
# Training loop with mini-batch SGD and step learning-rate decay
# ------------------------------------------------------------------
def train(X_train, y_train_oh, X_val, y_val,
          layer_sizes=(784, 128, 64, 10),
          epochs=20, batch_size=64,
          lr=0.1, lr_decay=0.5, decay_every=5,
          seed=42, verbose=True):

    rng = np.random.default_rng(seed)
    params = init_params(list(layer_sizes), seed=seed)

    N = X_train.shape[0]
    history = {"train_loss": [], "val_acc": [], "lr": []}
    current_lr = lr

    for epoch in range(1, epochs + 1):
        # Step-decay the learning rate every `decay_every` epochs
        if epoch > 1 and (epoch - 1) % decay_every == 0:
            current_lr *= lr_decay

        # Shuffle at the start of each epoch
        perm = rng.permutation(N)
        X_shuf = X_train[perm]
        y_shuf = y_train_oh[perm]

        epoch_losses = []
        for start in range(0, N, batch_size):
            end = start + batch_size
            Xb = X_shuf[start:end]
            yb = y_shuf[start:end]

            probs, cache = forward(Xb, params)
            loss = cross_entropy_loss(probs, yb)
            grads = backward(cache, params, yb)
            sgd_update(params, grads, current_lr)
            epoch_losses.append(loss)

        train_loss = float(np.mean(epoch_losses))
        val_acc = accuracy(X_val, y_val, params)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if verbose:
            print(f"Epoch {epoch:2d}  lr={current_lr:.4f}  "
                  f"train_loss={train_loss:.4f}  val_acc={val_acc*100:.2f}%")

    return params, history


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------
def plot_learning_curves(history, out_path="learning_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["train_loss"])
    ax1.set_title("Training loss")
    ax1.set_xlabel("Epoch")
    ax2.plot([v * 100 for v in history["val_acc"]])
    ax2.set_title("Validation accuracy (%)")
    ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, test_acc, out_path="confusion_matrix.png"):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — test acc {test_acc*100:.2f}%")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.show()


# ------------------------------------------------------------------
# Main entry point: train, evaluate, plot
# ------------------------------------------------------------------
if __name__ == "__main__":
    params_trained, history = train(
        X_train, y_train_one_hot, X_val, y_val,
        epochs=20, batch_size=64,
        lr=0.1, lr_decay=0.5, decay_every=5,
    )

    best_val = max(history["val_acc"])
    print(f"\nBest validation accuracy: {best_val*100:.2f}%")

    # Final test-set evaluation (run once, at the very end)
    test_acc = accuracy(X_test, y_test, params_trained)
    print(f"Test accuracy: {test_acc*100:.2f}%")

    # Save plots for the README
    plot_learning_curves(history)
    plot_confusion_matrix(y_test, predict(X_test, params_trained), test_acc)