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

ctypedef fused ndcomplexview_t:
    complex[::1]
    complex[:, ::1]
    complex[:, :, ::1]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef ndcomplexview_t _raw_fft(ndcomplexview_t x, bint forward, ndcomplexview_t out, size_t nthreads, size_t axis=0):
    if x is None:
        raise ValueError('x must not be None')
    if x.shape[axis] == 0:
        raise ValueError('len(x) must be larger 0')

    cdef size_t i
    if out is not None:
        for i in range(8):
            if out.shape[i] != x.shape[i]:
                raise ValueError(f'out and x must have same shape, but are have shapes {out.shape} and {x.shape}')

    cdef:
        double fct
        shape_t shape, axes
        stride_t stride_in
        stride_t stride_out
        complex * data_in
        complex * data_out
        size_t ndim

    if ndcomplexview_t is complex[::1]:
        ndim = 1
        data_in = &x[0]
        if out is None:
            out = np.empty(x.shape[0], dtype=complex)
        data_out = &out[0]
    elif ndcomplexview_t is complex[:, ::1]:
        ndim = 2
        data_in = &x[0, 0]
        if out is None:
            out = np.empty((x.shape[0], x.shape[1]), dtype=complex)
        data_out = &out[0, 0]
    elif ndcomplexview_t is complex[:, :, ::1]:
        ndim = 3
        data_in = &x[0, 0, 0]
        if out is None:
            out = np.empty((x.shape[0], x.shape[1], x.shape[2]), dtype=complex)
        data_out = &out[0, 0, 0]

    if axis >= ndim:
        raise ValueError(f'axis ({axis}) must not be larger or equal x.ndim ({ndim})')

    # initialize std::vectors
    for i in range(ndim):
        shape.push_back(x.shape[i])
        stride_in.push_back(x.strides[i])
        stride_out.push_back(out.strides[i])

    if forward:
        fct = 1.
    else:
        fct = 1. / x.shape[axis]

    axes.push_back(axis)
    c2c(shape, stride_in, stride_out, axes, forward, data_in, data_out, fct, nthreads)

    return out


cpdef ndcomplexview_t fft(ndcomplexview_t x, ndcomplexview_t out=None, size_t axis=0, size_t nthreads = 1):
    cdef bint forward = True
    return _raw_fft(x, forward, out, nthreads, axis)


cpdef ndcomplexview_t ifft(ndcomplexview_t x, ndcomplexview_t out=None, size_t axis=0, size_t nthreads = 1):
    cdef bint forward = False
    return _raw_fft(x, forward, out, nthreads, axis)