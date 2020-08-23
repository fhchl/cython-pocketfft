from libcpp.vector cimport vector

ctypedef vector[size_t] shape_t
ctypedef vector[ptrdiff_t] stride_t

ctypedef fused ndcomplexview_t:
    complex[::1]
    complex[:, ::1]
    complex[:, :, ::1]

cpdef ndcomplexview_t fft(ndcomplexview_t x, ndcomplexview_t out=*, size_t axis=*, size_t nthreads=*)
cpdef ndcomplexview_t ifft(ndcomplexview_t x, ndcomplexview_t out=*, size_t axis=*, size_t nthreads=*)