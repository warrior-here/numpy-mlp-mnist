# @title check.py
import sys, numpy as np
print(sys.executable)
print(np.__version__)
print(np.arange(6).reshape(2, 3) @ np.ones(3))