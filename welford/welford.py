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
            self.__m = np.mean(data, axis=0)
            self.__s = np.var(data, axis=0, ddof=0) * data.shape[0]

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
        return self.__getvars(ddof=1)

    @property
    def var_p(self):
        """Population variance of the recorded values"""
        return self.__getvars(ddof=0)

    def add(self, data, backup_flg=True):
        # argument check
        assert data.ndim == 1
        assert data.shape[0] == self.__dim

        # backup for rollbacking
        if backup_flg:
            self.__backup_attrs()

        # Welford's algorithm
        self.__count += 1
        delta = data - self.__m
        self.__m += delta / self.__count
        self.__s += delta * (data - self.__m)

    def add_all(self, data, backup_flg=True):
        # argument check
        assert data.ndim == 2
        assert data.shape[1] == self.__dim

        # backup for rollbacking
        if backup_flg:
            self.__backup_attrs()

        for elem in data:
            self.add(elem, backup_flg=False)

    def rollback(self):
        self.__count = self.__count_old
        self.__m = self.__m_old
        self.__s = self.__s_old

    def merge(self, other, backup_flg=True):
        """Merge this accumulator with another one."""
        # backup for rollbacking
        if backup_flg:
            self.__backup_attrs()
        count = self.__count + other.__count
        delta = self.__m - other.__m
        delta2 = delta * delta
        m = (self.__count * self.__m + other.__count * other.__m) / count
        s = self.__s + other.__s + delta2 * (self.__count * other.__count) / count
        self.__count = count
        self.__m = m
        self.__s = s

    def __getvars(self, ddof):
        min_count = ddof
        if self.__count <= min_count:
            return np.full(self.__dim, np.nan)
        else:
            return self.__s / (self.__count - ddof)

    def __backup_attrs(self):
        self.__count_old = self.__count
        self.__m_old = self.__m.copy()
        self.__s_old = self.__s.copy()
