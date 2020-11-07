Welford
=======

Python (Numpy) implementation of Welford’s algorithm, which is online or
parallel algorithm for calculating variances.

The algorithm is described in the followings,

-  `Wikipedia:Welford Online
   Algorithm <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm>`__
-  `Wikipedia:Welford Parallel
   Algorithm <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm>`__

Welford’s method is more numerically stable than the standard method.
The theoritical background of Welford’s method is mentioned in detail on
the following blog articles. Please refer them if you are interested in.

-  http://www.johndcook.com/blog/standard_deviation
-  https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/

This library is inspired by the `jvf’s
implementation <https://github.com/jvf/welford>`__, which is implemented
without using numpy library.

Install
-------

Download package via `PyPI
repository <https://pypi.org/project/welford/>`__

::

   $ pip install welford

Example
-------

For Online Calculation
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

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

Welford object supports initialization with data samples and batch
addition of samples.

.. code:: python

   # Initialize Welford object with samples
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

For Parallel Calculation
~~~~~~~~~~~~~~~~~~~~~~~~

Welford also offers parallel calculation method for variance.

.. code:: python

   import numpy as np
   from welford import Welford

   # Initialize two Welford objects
   w_1 = Welford()
   w_2 = Welford()

   # Each object will calculate variance of each samples in parallel.
   # On w_1
   w_1.add(np.array([0, 100]))
   w_1.add(np.array([1, 110]))
   w_1.add(np.array([2, 120]))
   print(w_1.var_s)  # sample variance -->[  1. 100.]
   print(w_1.var_p)  # population variance -->[ 0.66666667 66.66666667]

   # On w_2
   w_2.add(np.array([3, 130]))
   w_2.add(np.array([4, 140]))
   print(w_2.var_s)  # sample variance -->[ 0.5 50. ]
   print(w_2.var_p)  # sample variance -->[ 0.25 25.  ]

   # You can Merge objects to get variance of WHOLE samples
   w_1.merge(w_2)
   print(w.var_s)  # sample variance --> [  2.5 250. ]
   print(w_1.var_p)  # sample variance -->[  2. 200.]
