# distutils: language = c++
# cython: language_level = 3

from cython.operator cimport dereference as deref
import numpy as np

cimport cython


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

# TODO: Check github issue

# ctypedef fused ndcomplexview_t:
#     complex[::1]
#     complex[:, ::1]
#     complex[:, :, ::1]

def actual_cinit(FFT self, ndcomplexview_t arr):
    if ndcomplexview_t is complex[::1]:
        self.data = &arr[0]
    elif ndcomplexview_t is complex[:, ::1]:
        self.data = &arr[0, 0]
    elif ndcomplexview_t is complex[:, :, ::1]:
        self.data = &arr[0, 0, 0]

cdef class FFT:
    cdef complex *data
    def __cinit__(self, arr):
        actual_cinit(self, arr)

# cdef class FFT():
#     cdef:
#         shape_t *shape_ptr
#         stride_t *stride_in_ptr
#         stride_t *stride_out_ptr
#         shape_t *axes_ptr
#         complex *data_in_ptr
#         complex *data_out_ptr
#         double fct
#         bint forward

#     def __cinit__(self, ndcomplexview_t arr_in, size_t axis, bint forward, ndcomplexview_t arr_out):

#         if arr_in is None or arr_out is None:
#             raise ValueError('arr_in and arr_out must not be None')
#         if arr_in.shape[axis] == 0:
#             raise ValueError('arr_in.shape[axis] must be larger 0')

#         cdef size_t i
#         for i in range(8):
#             if arr_out.shape[i] != arr_in.shape[i]:
#                 raise ValueError(f'arr_out and arr_in must have same shape, but have shapes {arr_out.shape} and {arr_in.shape}')

#         self.shape_ptr = new shape_t()
#         self.axes_ptr = new shape_t()
#         self.stride_in_ptr = new stride_t()
#         self.stride_out_ptr = new stride_t()

#         cdef size_t ndim
#         if ndcomplexview_t is complex[::1]:
#             ndim = 1
#             self.data_in_ptr = &arr_in[0]
#             self.data_out_ptr = &arr_out[0]
#         elif ndcomplexview_t is complex[:, ::1]:
#             ndim = 2
#             self.data_in_ptr = &arr_in[0, 0]
#             self.data_out_ptr = &arr_out[0, 0]
#         elif ndcomplexview_t is complex[:, :, ::1]:
#             ndim = 3
#             self.data_in_ptr = &arr_in[0, 0, 0]
#             self.data_out_ptr = &arr_out[0, 0, 0]

#         if axis >= ndim:
#             raise ValueError(f'axis ({axis}) must not be larger or equal arr_in.ndim ({ndim})')

#         # initialize std::vectors
#         for i in range(ndim):
#             self.shape_ptr.push_back(arr_in.shape[i])
#             self.stride_in_ptr.push_back(arr_in.strides[i])
#             self.stride_out_ptr.push_back(arr_out.strides[i])

#         self.axes_ptr.push_back(axis)
#         self.forward = forward

#         if forward:
#             fct = 1.
#         else:
#             fct = 1. / arr_in.shape[axis]

#     def __dealloc__(self):
#         del self.shape_ptr, self.stride_in_ptr, self.stride_out_ptr, self.axes_ptr

#     cdef run(FFT self, size_t nthreads=1):
#         c2c(
#             deref(self.shape_ptr),
#             deref(self.stride_in_ptr),
#             deref(self.stride_out_ptr),
#             deref(self.axes_ptr),
#             self.forward,
#             self.data_in_ptr,
#             self.data_out_ptr,
#             self.fct,
#             nthreads
#         )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef ndcomplexview_t _fft(ndcomplexview_t x, bint forward, ndcomplexview_t out, size_t nthreads, size_t axis=0):
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
        complex *data_in
        complex *data_out
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
    return _fft(x, forward, out, nthreads, axis)


cpdef ndcomplexview_t ifft(ndcomplexview_t x, ndcomplexview_t out=None, size_t axis=0, size_t nthreads = 1):
    cdef bint forward = False
    return _fft(x, forward, out, nthreads, axis)