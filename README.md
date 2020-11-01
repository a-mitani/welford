# Welford
Python (Numpy) implementation of Welford's algorithm,
which is online or parallel algorithm for calculating variances.

Welford's method is more numerically stable than the standard method. The theoritical background of Welford's method is mentioned in detail on the following blog articles. Please refer them if you are interested in.

* www.johndcook.com/blog/standard_deviation
* jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/

This library is inspired by the jvf's implementation, which is implemented without using numpy library. 
* implementaion done by jvf: github.com/jvf/welford

## Install
```
$ pip install welford
```

## Example
### for online calculation
```python
import numpy as np
from welford import Welford

# Initialize Welford object
w = Welford()

# Input data samples sequentialy
w.add(np.array([0, 100]))
w.add(np.array([1, 110]))
w.add(np.array([2, 120]))

# output
print(w.mean)  # mean --> [  1. 110.]
print(w.var_s)  # sample variance --> [1, 100]
print(w.var_p)  # population variance --> [ 0.6666 66.66]

# You can add other samples after calculating variances.
w.add(np.array([3, 130]))
w.add(np.array([4, 140]))

# output with added samples
print(w.mean)  # mean --> [  2. 120.]
print(w.var_s)  # sample variance --> [  2.5 250. ]
print(w.var_p)  # population variance --> [  2. 200.]
```

Welford object supports initialization with data samples and batch addition of samples.
```python
# Initialize Welford object
ini = np.array([[0, 100], 
                [1, 110], 
                [2, 120]])
w = Welford(ini)

# output
print(w.mean)  # mean --> [  1. 110.]
print(w.var_s)  # sample variance --> [1, 100]
print(w.var_p)  # population variance --> [ 0.66666667 66.66666667]

# add other samples through batch method
other_samples = np.array([[3, 130], 
                          [4, 140]])
w.add_all(other_samples)

# output with added samples
print(w.mean)  # mean --> [  2. 120.]
print(w.var_s)  # sample variance --> [  2.5 250. ]
print(w.var_p)  # population variance --> [  2. 200.]
```