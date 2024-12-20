# PyDkMIPS

A Python package for the DkMIPS algorithm, providing efficient implementation through C++ bindings.

Build on top of [DkMIPS-api](https://github.com/HuangQiang/DkMIPS-api).

## Installation

You can install the package using pip:

```bash
pip install .
```

## Requirements

- Python >= 3.6
- NumPy >= 1.19.0
- pybind11 >= 2.6.0
- C++ compiler with C++11 support
- OpenMP support

## Usage

```python
import numpy as np
from pydkmips

# Initialize with n items and d dimensions
n, d = 1000, 128
items = np.random.rand(n, d).astype(np.float32)
index = pydkmips.BC_Greedy(n) # if you only have one single space
# index = pydkmips.BC_Greedy(n1, n2) # if you have two spaces

index.add(items) # if you only have one single space
# index.add(items1, items2) # if you have two spaces

# Run DkMIPS+ algorithm
k = 10  # number of items to retrieve
lambda_param = 0.5
c = 0.5 # this parameter need to be tuned
query = np.random.rand(d).astype(np.float32)
results = index.search(query, k, lambda_param, c, objective="avg") # if you want to use max objective, just change "avg" to "max"
```