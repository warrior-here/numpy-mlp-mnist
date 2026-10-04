# @title dataloader.py
import numpy as np
import gzip

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float32) / 255.0

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

# Load the training data
X_train = load_images('data/train-images-idx3-ubyte.gz')
y_train = load_labels('data/train-labels-idx1-ubyte.gz')

# Load the test data
X_test  = load_images('data/t10k-images-idx3-ubyte.gz')
y_test  = load_labels('data/t10k-labels-idx1-ubyte.gz')

# Split the training data into train and validation sets
X_train, X_val = X_train[:50000], X_train[50000:]
y_train, y_val = y_train[:50000], y_train[50000:]

# One-hot encode the labels
y_train_one_hot = one_hot(y_train)
y_val_one_hot = one_hot(y_val)
y_test_one_hot = one_hot(y_test)

print(f"Train: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
print(f"Val: X_val.shape={X_val.shape}, y_val.shape={y_val.shape}")
print(f"Test: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

import matplotlib.pyplot as plt

plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()