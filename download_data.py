# @title download_data.py
import os, urllib.request

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

os.makedirs("data", exist_ok=True)
for name in FILES:
    dest = os.path.join("data", name)
    if os.path.exists(dest):
        continue
    for base in MIRRORS:
        try:
            urllib.request.urlretrieve(base + name, dest)
            print("downloaded", name)
            break
        except Exception as e:
            print("failed from", base, "-", e)