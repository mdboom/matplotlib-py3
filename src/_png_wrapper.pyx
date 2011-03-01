cimport numpy as np
import numpy as np

np.import_array()

cdef extern from "_png.h":
    cdef object _write_png 'write_png' (object buff, int width, int height, object fileobj,
                                        double dpi)
    cdef object _read_png_float 'read_png_float' (object fileobj)
    cdef object _read_png_uint8 'read_png_uint8' (object fileobj)

def write_png(buff, width, height, fileobj, dpi = 0.0):
    return _write_png(buff, width, height, fileobj, dpi)

def read_png(fileobj):
    return _read_png_float(fileobj)
read_png_float = read_png

def read_png_uint8(fileobj):
    return _read_png_uint8(fileobj)
