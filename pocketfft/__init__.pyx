# distutils: language = c++
# cython: language_level = 3

from cython.operator cimport dereference as deref
cimport numpy as np
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


def __complexFFT_actual_cinit(ComplexFFT self, ndcomplexview_t arr_in, ndcomplexview_t arr_out):
    """Workaround for Cython issue with templated argunments in __cinit__."""

    if ndcomplexview_t is complex[::1]:
        self.ndim = 1
        self.data_in = &arr_in[0]
        self.data_out = &arr_out[0]
    elif ndcomplexview_t is complex[:, ::1]:
        self.ndim = 2
        self.data_in = &arr_in[0, 0]
        self.data_out = &arr_out[0, 0]
    elif ndcomplexview_t is complex[:, :, ::1]:
        self.ndim = 3
        self.data_in = &arr_in[0, 0, 0]
        self.data_out = &arr_out[0, 0, 0]

    # initialize std::vectors
    self.shape = new shape_t()
    self.axes = new shape_t()
    self.stride_in = new stride_t()
    self.stride_out = new stride_t()

    for i in range(self.ndim):
        self.shape.push_back(arr_in.shape[i])
        self.stride_in.push_back(arr_in.strides[i])
        self.stride_out.push_back(arr_out.strides[i])

    self.axes.push_back(self.axis)

    self.fct_forward = 1.
    self.fct_backward = 1. / arr_in.shape[self.axis]


cdef class ComplexFFT:
    cdef:
        shape_t *shape
        shape_t *axes
        stride_t *stride_in
        stride_t *stride_out
        complex *data_in
        complex *data_out
        double fct_forward
        double fct_backward
        size_t ndim
        size_t axis
        size_t nsamp
        int nthreads

    def __cinit__(self, arr_in, arr_out, nthreads=1, axis=-1):
        self.nthreads = nthreads

        # check shapes
        if arr_in.shape != arr_out.shape:
            raise ValueError(f'arrays must have same shape but have shapes {arr_in.shape, arr_out.shape}')
        self.ndim = arr_in.ndim

        if arr_in.shape[axis] == 0:
            raise ValueError('len(x) must be larger 0')

        self.axis = self.ndim + axis if axis < 0 else axis

        if not (arr_in.dtype == arr_out.dtype == np.complex):
            raise ValueError('arrays must have complex dtype')

        __complexFFT_actual_cinit(self, arr_in, arr_out)

    def __dealloc__(ComplexFFT self):
        del self.shape, self.stride_in, self.stride_out, self.axes

    cpdef void forward(ComplexFFT self):
        c2c(
            deref(self.shape),
            deref(self.stride_in),
            deref(self.stride_out),
            deref(self.axes),
            True,
            self.data_in,
            self.data_out,
            self.fct_forward,
            self.nthreads
        )

    cpdef void backward(ComplexFFT self):
        c2c(
            deref(self.shape),
            deref(self.stride_in),
            deref(self.stride_out),
            deref(self.axes),
            False,
            self.data_in,
            self.data_out,
            self.fct_backward,
            self.nthreads
        )


def __realFFT_actual_cinit(realFFT self, nddoubleview_t arr_in, ndcomplexview_t arr_out):
    """Workaround for Cython issue with templated argunments in __cinit__."""

    if nddoubleview_t is double[::1]:
        self.ndim = 1
        self.data_in = &arr_in[0]
    elif nddoubleview_t is double[:, ::1]:
        self.ndim = 2
        self.data_in = &arr_in[0, 0]
    elif nddoubleview_t is double[:, :, ::1]:
        self.ndim = 3
        self.data_in = &arr_in[0, 0, 0]

    if ndcomplexview_t is complex[::1]:
        self.data_out = &arr_out[0]
    elif ndcomplexview_t is complex[:, ::1]:
        self.data_out = &arr_out[0, 0]
    elif ndcomplexview_t is complex[:, :, ::1]:
        self.data_out = &arr_out[0, 0, 0]

    # initialize std::vectors
    self.shape_in = new shape_t()
    self.stride_in = new stride_t()
    self.stride_out = new stride_t()

    for i in range(self.ndim):
        self.shape_in.push_back(arr_in.shape[i])
        self.stride_in.push_back(arr_in.strides[i])
        self.stride_out.push_back(arr_out.strides[i])

    self.fct_forward = 1.
    self.fct_backward = 1. / arr_in.shape[self.axis]


cdef class realFFT:
    cdef:
        shape_t *shape_in
        stride_t *stride_in
        stride_t *stride_out
        double *data_in
        complex *data_out
        double fct_forward
        double fct_backward
        size_t ndim
        size_t axis
        size_t nsamp
        int nthreads

    def __cinit__(self, arr_in, arr_out, nthreads=1, axis=-1):
        self.nthreads = nthreads

        # check shapes
        if arr_in.ndim != arr_out.ndim:
            raise ValueError(f'arrays must have same number of dimensions but have shapes {arr_in.shape, arr_out.shape}')
        self.ndim = arr_in.ndim

        if not (- arr_in.ndim <= axis < self.ndim):
            raise ValueError('axis invalid')
        self.axis = self.ndim + axis if axis < 0 else axis

        for i in range(self.ndim):
            if i == self.axis:
                if arr_out.shape[i] != arr_in.shape[i]//2 + 1:
                    raise ValueError(f'incompatible shapes {arr_in.shape, arr_out.shape} at dim {i}')
            else:
                if arr_out.shape[i] != arr_in.shape[i]:
                    raise ValueError(f'incompatible shapes {arr_in.shape, arr_out.shape} at dim {i}')

        if arr_in.shape[axis] == 0:
            raise ValueError('dimensions must not be 0')

        if not (arr_in.dtype == float and arr_out.dtype == complex):
            raise ValueError('arrays must have dtype double and complex but have {arr_in.dtype, arr_out.dtype}')

        __realFFT_actual_cinit(self, arr_in, arr_out)

    def __dealloc__(realFFT self):
        del self.shape_in, self.stride_in, self.stride_out

    cpdef void forward(realFFT self):
        r2c(
            deref(self.shape_in),
            deref(self.stride_in),
            deref(self.stride_out),
            self.axis,
            True,
            self.data_in,
            self.data_out,
            self.fct_forward,
            self.nthreads
        )

    cpdef void backward(realFFT self):
        c2r(
            deref(self.shape_in),
            deref(self.stride_out),
            deref(self.stride_in),
            self.axis,
            False,
            self.data_out,
            self.data_in,
            self.fct_backward,
            self.nthreads
        )


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