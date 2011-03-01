#ifndef ___PNG_H__
#define ___PNG_H__

#include "Python.h"

#ifdef __cplusplus
extern "C" {
#endif

PyObject*
write_png(PyObject* buffer, Py_ssize_t width, Py_ssize_t height,
          PyObject* py_fileobj, double dpi);

PyObject*
read_png_float(PyObject* py_file_obj);

PyObject*
read_png_uint8(PyObject* py_file_obj);

#ifdef __cplusplus
}
#endif

#endif
