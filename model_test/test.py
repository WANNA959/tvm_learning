import numpy as np

nums = np.array([2, 1, 3])
idx = np.argsort(nums)[::-1]
print(idx)
