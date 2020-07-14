# distutils: language = c++
# cython: language_level = 3

from libcpp.vector cimport vector
import numpy as np

cimport cython

ctypedef vector[size_t] shape_t
ctypedef vector[ptrdiff_t] stride_t


cdef extern from "pocketfft_hdronly.h" namespace "pocketfft":
    void c2c(const shape_t &shape, const stride_t &stride_in,
        const stride_t &stride_out, const shape_t &axes, bint forward,
        const complex *data_in, complex *data_out, double fct,
        size_t nthreads)

    void r2c(const shape_t &shape_in, const stride_t &stride_in,
        const stride_t &stride_out, size_t axis, bint forward,
        const double *data_in, complex *data_out, double fct, size_t nthreads)

    void c2r(const shape_t &shape_out, const stride_t &stride_in,
        const stride_t &stride_out, size_t axis, bint forward,
        const complex *data_in, double *data_out, double fct, size_t nthreads)

@cython.boundscheck(False)
cdef complex[:] _raw_fft(complex[:] x, bint forward, double fct, complex[:] out, size_t nthreads):
    if x is None:
        raise ValueError('x must not be None')

    cdef size_t n = x.shape[0] # TODO: make this dynamic

    if n == 0:
        raise ValueError('len(x) must be larger 0')
    if out is None:
        out = np.empty(n, dtype=complex)


    cdef:
        shape_t shape, axes
        stride_t stride_in
        stride_t stride_out
        complex *data_in = &x[0]
        complex *data_out = &out[0]

    shape.push_back(n)
    stride_in.push_back(x.strides[0])
    stride_out.push_back(out.strides[0])
    axes.push_back(0)

    c2c(shape, stride_in, stride_out, axes, forward, data_in, data_out, fct, nthreads)

    return out


cpdef complex[:] fft(complex[:] x, complex[:] out=None, size_t nthreads = 1):
    cdef bint forward = True
    cdef double fct = 1.
    return _raw_fft(x, forward, fct, out, nthreads)


cpdef complex[:] ifft(complex[:] x, complex[:] out=None, size_t nthreads = 1):
    cdef bint forward = False
    cdef double fct = 1. / x.shape[0]
    return _raw_fft(x, forward, fct, out, nthreads)


cdef complex[:] rfft(double[:] x, int n=0):
    pass


cdef double[:] irfft(complex[:] x, int n=0):
    pass
