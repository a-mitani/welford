#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This library is python(numpy) implementation of Welford's algorithm, 
which is online and parallel algorithm for calculating variances.

Welfords method is more numerically stable than the standard method as
described in the following blog,
    * Accurately computing running variance: www.johndcook.com/blog/standard_deviation

This library is inspired by the jvf's implementation, which is implemented
without using numpy library.
    * implementaion done by jvf: github.com/jvf/welford
"""
import numpy as np


class Welford:
    """Accumulator object for Welfords online variance algorithm."""

    def __init__(self, data=None, dim=None):
        """Initialize with an optional data."""
        # Check arguments
        if data is None:
            if dim is None:
                raise AttributeError("If data is not assigned, 'dim' must be assigned")
        else:
            assert data.ndim == 2
            if dim is not None:
                assert Nonedata.shape[1] == dim

        # Initialize instance attributes
        if data is not None:
            self.__dim = data.shape[1]
        else:
            self.__dim = dim

        if data is None:
            self.__count = 0
            self.__m = np.zeros(self.__dim)
            self.__s = np.zeros(self.__dim)
        else:
            self.__count = data.shape[0]
            self.__m = np.mean(data)
            self.__s = np.var(data, ddof=0) * data.shape[0]

        # previous attribute values for rollbacking
        self.__count_old = None
        self.__m_old = None
        self.__s_old = None

    @property
    def count(self):
        """The number of recorded values"""
        return self.__count

    @property
    def mean(self):
        """Mean of the recorded values"""
        return self.__m

    @property
    def var_s(self):
        """Sample variance of the recorded values"""
        return self.__getvars(ddof=0)

    @property
    def var_p(self):
        """Population variance of the recorded values"""
        return self.__getvars(ddof=1)

    def add_all(self, data):
        pass

    def add(self, data):
        pass

    def merge(self, other):
        pass

    def __getvars(self, ddof):
        min_count = ddof
        if self.__count <= min_count:
            return np.full(self.__dim, np.nan)
        else:
            return self.__s / (self.__count - ddof)
