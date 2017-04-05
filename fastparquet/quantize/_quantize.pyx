import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free, realloc
from libc.math cimport isnan, NAN

np.import_array()

ctypedef np.float32_t DTYPE_FLOAT32
ctypedef np.float64_t DTYPE_FLOAT64
ctypedef np.npy_intp DTYPE_NPY_INTP

cdef extern from "rice.h" nogil:
    int rcomp(int *input, int nx, unsigned char *buf, int maxbuflen, int nblock)
    int rdecomp(unsigned char *buf, int buflen, int *output, int nx, int nblock)
    int NINT(double x)

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef extern from "Python.h":
    object PyBytes_FromStringAndSize(const char *v, Py_ssize_t len)
    char *PyBytes_AsString(object o)

RICE_MESSAGES = {-1: "end of buffer encountered!",
                 -2: "out of memory!"}


class FloatQuantizationError(Exception):
    def __init__(self, msg=None):
        super().__init__(self, msg)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef quantize_and_compress_float32(float [::1] data,
                                    double scale,
                                    double zero_point,
                                    long seed,
                                    long n_dithers=10000,
                                    int blocksize=32):
    """
    Perform float quantization and then rice compression on float 32 data.
    """
    cdef:
        int n_data = data.size
        int max_buff_len = data.size * 4
        int *idata;
        unsigned char *buff
        unsigned char *c
        double *dbuff
        long *lbuff
        int *head
        int blen
        int i
        int dloc = 0
        double [:] dithers

    rng = np.random.RandomState(seed=seed)
    dithers = rng.uniform(size=n_dithers)

    with nogil:
        idata = <int*>malloc(n_data * sizeof(int))
        buff = <unsigned char*>malloc(
            max_buff_len * sizeof(unsigned char) + 2 * sizeof(double) + 3 * sizeof(long) + 8)
        dbuff = <double*>buff
        lbuff = <long*>(buff + 16)
        head = <int*>(buff + 40)
        c = <unsigned char*>(buff + 48)
        dbuff[0] = scale
        dbuff[1] = zero_point
        lbuff[0] = seed
        lbuff[1] = n_dithers
        lbuff[2] = 4
        head[0] = n_data
        head[1] = blocksize

        for i from 0 <= i < n_data by 1:
            if data[i] == 0.0:
                idata[i] = -2147483645
            elif data[i] == 1.0:
                idata[i] = -2147483646
            elif isnan(data[i]):
                idata[i] = -2147483647
            else:
                idata[i] = NINT(((<double> data[i]) - zero_point) / scale + dithers[dloc] - 0.5)
            dloc += 1
            if dloc >= n_dithers:
                dloc = 0

        blen = rcomp(idata, n_data, c, max_buff_len, blocksize)
        free(idata)

    if blen <= 0:
        free(buff)
        raise FloatQuantizationError("compression: " + RICE_MESSAGES[blen])

    with nogil:
        buff = <unsigned char*>realloc(buff, blen + 48)

    return PyBytes_FromStringAndSize(<char *>buff, blen + 48)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef quantize_and_compress_float64(double [::1] data,
                                    double scale,
                                    double zero_point,
                                    long seed,
                                    long n_dithers=10000,
                                    int blocksize=32):
    """
    Perform float quantization and then rice compression on float 64 data.
    """
    cdef:
        int n_data = data.size
        int max_buff_len = data.size * 4
        int *idata;
        unsigned char *buff
        unsigned char *c
        double *dbuff
        long *lbuff
        int *head
        int blen
        int i
        int dloc = 0
        double [:] dithers

    rng = np.random.RandomState(seed=seed)
    dithers = rng.uniform(size=n_dithers)

    with nogil:
        idata = <int*>malloc(n_data * sizeof(int))
        buff = <unsigned char*>malloc(
            max_buff_len * sizeof(unsigned char) + 2 * sizeof(double) + 3 * sizeof(long) + 8)
        dbuff = <double*>buff
        lbuff = <long*>(buff + 16)
        head = <int*>(buff + 40)
        c = <unsigned char*>(buff + 48)
        dbuff[0] = scale
        dbuff[1] = zero_point
        lbuff[0] = seed
        lbuff[1] = n_dithers
        lbuff[2] = 8
        head[0] = n_data
        head[1] = blocksize

        for i from 0 <= i < n_data by 1:
            if data[i] == 0.0:
                idata[i] = -2147483645
            elif data[i] == 1.0:
                idata[i] = -2147483646
            elif isnan(data[i]):
                idata[i] = -2147483647
            else:
                idata[i] = NINT((data[i] - zero_point) / scale + dithers[dloc] - 0.5)
            dloc += 1
            if dloc >= n_dithers:
                dloc = 0

        blen = rcomp(idata, n_data, c, max_buff_len, blocksize)
        free(idata)

    if blen <= 0:
        free(buff)
        raise FloatQuantizationError("compression: " + RICE_MESSAGES[blen])

    with nogil:
        buff = <unsigned char*>realloc(buff, blen + 48)

    return PyBytes_FromStringAndSize(<char *>buff, blen + 48)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dequantize_and_decompress(bytes _buff):
    """
    Perform float de-quanization and rice decompression to float 32 data.
    """
    cdef:
        double scale
        double zero_point
        long seed
        long n_dithers
        long itemsize
        int n_data
        int blocksize
        DTYPE_NPY_INTP n_data_intp

        unsigned char *c
        unsigned char *buff
        int ret
        void *data
        float *fdata
        double *ddata
        int *idata
        int buff_len
        int i
        int dloc = 0
        double [:] dithers
        np.ndarray[DTYPE_FLOAT32, ndim=1] farr
        np.ndarray[DTYPE_FLOAT64, ndim=1] darr

    buff = <unsigned char*>PyBytes_AsString(_buff)
    buff_len = len(_buff) - 48

    with nogil:
        dbuff = <double*>buff
        lbuff = <long*>(buff + 16)
        head = <int*>(buff + 40)
        c = <unsigned char*>(buff + 48)
        scale = dbuff[0]
        zero_point = dbuff[1]
        seed = lbuff[0]
        n_dithers = lbuff[1]
        itemsize = lbuff[2]
        n_data = head[0]
        blocksize = head[1]

        if itemsize == 4:
            data = <void*>malloc(n_data * sizeof(float))
            fdata = <float*>data
        else:
            data = <void*>malloc(n_data * sizeof(double))
            ddata = <double*>data

        idata = <int*>malloc(n_data * sizeof(int))
        ret = rdecomp(c, buff_len, idata, n_data, blocksize)

    if ret == 0:
        rng = np.random.RandomState(seed=seed)
        dithers = rng.uniform(size=n_dithers)
    else:
        free(idata)
        free(data)
        raise FloatQuantizationError("decompression: " + RICE_MESSAGES[ret])

    with nogil:
        if itemsize == 4:
            for i from 0 <= i < n_data by 1:
                if idata[i] == -2147483645:
                    fdata[i] = 0.0
                elif idata[i] == -2147483646:
                    fdata[i] = 1.0
                elif idata[i] == -2147483647:
                    fdata[i] = NAN
                else:
                    fdata[i] = <float>(((idata[i] - dithers[dloc] + 0.5) * scale) + zero_point)
                dloc += 1
                if dloc >= n_dithers:
                    dloc = 0
        else:
            for i from 0 <= i < n_data by 1:
                if idata[i] == -2147483645:
                    ddata[i] = 0.0
                elif idata[i] == -2147483646:
                    ddata[i] = 1.0
                elif idata[i] == -2147483647:
                    ddata[i] = NAN
                else:
                    ddata[i] = <double>(((idata[i] - dithers[dloc] + 0.5) * scale) + zero_point)
                dloc += 1
                if dloc >= n_dithers:
                    dloc = 0

        free(idata)

    n_data_intp = n_data
    if itemsize == 4:
        farr = np.PyArray_SimpleNewFromData(1, &n_data_intp, np.NPY_FLOAT32, data)
        PyArray_ENABLEFLAGS(farr, np.NPY_OWNDATA)
        return farr
    else:
        darr = np.PyArray_SimpleNewFromData(1, &n_data_intp, np.NPY_FLOAT64, data)
        PyArray_ENABLEFLAGS(darr, np.NPY_OWNDATA)
        return darr
