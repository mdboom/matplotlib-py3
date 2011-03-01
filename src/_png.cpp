/* -*- mode: c++; c-basic-offset: 4 -*- */

/* For linux, png.h must be imported before Python.h because
   png.h needs to be the one to define setjmp.
   Undefining _POSIX_C_SOURCE and _XOPEN_SOURCE stops a couple
   of harmless warnings.
*/

#ifdef __linux__
#   include <png.h>
#   ifdef _POSIX_C_SOURCE
#       undef _POSIX_C_SOURCE
#   endif
#   ifdef _XOPEN_SOURCE
#       undef _XOPEN_SOURCE
#   endif
#   include "Python.h"
#else

/* Python API mandates Python.h is included *first* */
#   include "Python.h"
#   include <png.h>
#endif

#if PY_MAJOR_VERSION >= 3
#define PY3K 1
#else
#define PY3K 0
#endif

#include "numpy/arrayobject.h"

// As reported in [3082058] build _png.so on aix
#ifdef _AIX
#undef jmpbuf
#endif

static void
write_png_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    PyObject* py_file_obj = (PyObject*)png_get_io_ptr(png_ptr);
    PyObject* write_method = PyObject_GetAttrString(py_file_obj, "write");
    PyObject* result = NULL;
    if (write_method)
    {
        #if PY3K
        result = PyObject_CallFunction(write_method, (char *)"y#", data,
                                       length);
        #else
        result = PyObject_CallFunction(write_method, (char *)"s#", data,
                                       length);
        #endif
    }
    Py_XDECREF(write_method);
    Py_XDECREF(result);
}

static void
flush_png_data(png_structp png_ptr)
{
    PyObject* py_file_obj = (PyObject*)png_get_io_ptr(png_ptr);
    PyObject* flush_method = PyObject_GetAttrString(py_file_obj, "flush");
    PyObject* result = NULL;
    if (flush_method)
    {
        result = PyObject_CallFunction(flush_method, (char *)"");
    }
    Py_XDECREF(flush_method);
    Py_XDECREF(result);
}

// this code is heavily adapted from the paint license, which is in
// the file paint.license (BSD compatible) included in this
// distribution.  TODO, add license file to MANIFEST.in and CVS
extern "C" PyObject*
write_png(PyObject* buffer, Py_ssize_t width, Py_ssize_t height,
          PyObject* py_fileobj, double dpi)
{
    FILE *fp = NULL;
    const char* filename = NULL;
    bool close_file = false;
    PyObject* result = NULL;
    const void* pixbufp = NULL;
    Py_ssize_t pixbuf_len = 0;
    png_byte* pixbuf = NULL;
    int fd = -1;

    png_bytep *row_pointers = NULL;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    struct png_color_8_struct sig_bit;
    png_uint_32 row;

    if (!PyObject_CheckReadBuffer(buffer))
    {
        PyErr_SetString(
            PyExc_TypeError,
            "First argument must be an rgba buffer.");
        goto exit;
    }

    if (PyObject_AsReadBuffer(buffer, &pixbufp, &pixbuf_len))
    {
        PyErr_SetString(
            PyExc_ValueError,
            "Couldn't get data from read buffer.");
        goto exit;
    }

    if (pixbuf_len < width * height * 4)
    {
        PyErr_SetString(
            PyExc_ValueError,
            "Buffer and width, height don't seem to match.");
        goto exit;
    }

    #if PY3K
    fd = PyObject_AsFileDescriptor(py_fileobj);
    PyErr_Clear();
    #endif

    #if PY3K
    if (PyBytes_Check(py_fileobj))
    {
        filename = PyBytes_AsString(py_fileobj);
    }
    #else
    if (PyString_Check(py_fileobj))
    {
        filename = PyString_AsString(py_fileobj);
    }
    #endif
    if (filename)
    {
        if ((fp = fopen(filename, "wb")) == NULL)
        {
            PyErr_Format(
                PyExc_IOError,
                "Could not open file '%s'",
                filename);
            goto exit;
        }
        close_file = true;
    }
    #if PY3K
    else if (fd != -1)
    {
        fp = fdopen(fd, "w");
    }
    #else
    else if (PyFile_CheckExact(py_fileobj))
    {
        fp = PyFile_AsFile(py_fileobj);
    }
    #endif
    else
    {
        PyObject* write_method = PyObject_GetAttrString(
            py_fileobj, "write");
        if (!(write_method && PyCallable_Check(write_method)))
        {
            Py_XDECREF(write_method);
            PyErr_SetString(
                PyExc_TypeError,
                "Object does not appear to be a 8-bit string path or a "
                "Python file-like object");
            goto exit;
        }
        Py_XDECREF(write_method);
    }

    pixbuf = (png_byte *)pixbufp;
    row_pointers = new png_bytep[height];
    for (row = 0; row < (png_uint_32)height; ++row)
    {
        row_pointers[row] = pixbuf + row * width * 4;
    }

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL)
    {
        PyErr_SetString(
            PyExc_RuntimeError, "Could not create write struct");
        goto exit;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        PyErr_SetString(
            PyExc_RuntimeError, "Could not create info struct");
        goto exit;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        PyErr_SetString(
            PyExc_RuntimeError, "Error building image");
        goto exit;
    }

    if (fp)
    {
        png_init_io(png_ptr, fp);
    }
    else
    {
        /* Write to a Python function */
        png_set_write_fn(png_ptr, (void*)py_fileobj,
                         &write_png_data, &flush_png_data);
    }
    png_set_IHDR(png_ptr, info_ptr,
                 width, height, 8,
                 PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    // Save the dpi of the image in the file
    if (dpi != 0.0)
    {
        size_t dots_per_meter = (size_t)(dpi / (2.54 / 100.0));
        png_set_pHYs(png_ptr, info_ptr, dots_per_meter, dots_per_meter,
                     PNG_RESOLUTION_METER);
    }

    // this a a color image!
    sig_bit.gray = 0;
    sig_bit.red = 8;
    sig_bit.green = 8;
    sig_bit.blue = 8;
    sig_bit.alpha = 8;
    png_set_sBIT(png_ptr, info_ptr, &sig_bit);

    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, info_ptr);

    Py_INCREF(Py_None);
    result = Py_None;

  exit:

    if (png_ptr && info_ptr)
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
    }
    delete [] row_pointers;
    #if PY3K
    if (fp)
    {
        fflush(fp);
    }
    #endif
    if (fp && close_file)
    {
        fclose(fp);
    }

    return result;
}

static void
_read_png_data(PyObject* py_file_obj, png_bytep data, png_size_t length)
{
    PyObject* read_method = PyObject_GetAttrString(py_file_obj, "read");
    PyObject* result = NULL;
    char *buffer;
    Py_ssize_t bufflen;
    if (read_method)
    {
        result = PyObject_CallFunction(read_method, (char *)"i", length);
    }

    #if PY3K
    if (PyBytes_AsStringAndSize(result, &buffer, &bufflen) == 0)
    #else
    if (PyString_AsStringAndSize(result, &buffer, &bufflen) == 0)
    #endif
    {
        if (bufflen == (Py_ssize_t)length)
        {
            memcpy(data, buffer, length);
        }
    }
    Py_XDECREF(read_method);
    Py_XDECREF(result);
}

static void
read_png_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    PyObject* py_file_obj = (PyObject*)png_get_io_ptr(png_ptr);
    _read_png_data(py_file_obj, data, length);
}

static PyObject*
_read_png(PyObject* py_fileobj, int result_type)
{
    png_byte header[8];   // 8 is the maximum size that can be checked
    FILE* fp = NULL;
    bool close_file = false;
    const char* filename = NULL;
    int fd = -1;
    PyArrayObject* A = NULL;
    PyObject* result = NULL;
    int num_dims;
    npy_intp dimensions[3];
    png_uint_32 width = 0;
    png_uint_32 height = 0;
    int bit_depth;

    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_bytep *row_pointers = NULL;
    png_uint_32 row;

    if (result_type != NPY_FLOAT && result_type != NPY_UBYTE) {
        PyErr_SetString(
            PyExc_ValueError,
            "Only 'float' and 'uint8' are allowed output types");
        return NULL;
    }

    #if PY3K
    fd = PyObject_AsFileDescriptor(py_fileobj);
    PyErr_Clear();
    #endif

    #if PY3K
    if (PyBytes_Check(py_fileobj)) {
        filename = PyBytes_AsString(py_fileobj);
    }
    #else
    if (PyString_Check(py_fileobj)) {
        filename = PyString_AsString(py_fileobj);
    }
    #endif

    if (filename)
    {
        if ((fp = fopen(filename, "rb")) == NULL)
        {
            PyErr_Format(
                PyExc_IOError,
                "Could not open file '%s' for reading",
                filename);
            goto exit;
        }
        close_file = true;
    }
    #if PY3K
    else if (fd != -1) {
        fp = fdopen(fd, "rb");
    }
    #else
    else if (PyFile_CheckExact(py_fileobj.ptr()))
    {
        fp = PyFile_AsFile(py_fileobj.ptr());
    }
    #endif
    else
    {
        PyObject* read_method = PyObject_GetAttrString(py_fileobj, "read");
        if (!(read_method && PyCallable_Check(read_method)))
        {
            Py_XDECREF(read_method);
            PyErr_SetString(
                PyExc_TypeError,
                "Object does not appear to be a 8-bit string path or a "
                "Python file-like object");
            goto exit;
        }
        Py_XDECREF(read_method);
    }

    if (fp)
    {
        if (fread(header, 1, 8, fp) != 8)
        {
            PyErr_SetString(
                PyExc_IOError,
                "Error reading PNG header");
            goto exit;
        }
    }
    else
    {
        _read_png_data(py_fileobj, header, 8);
    }

    if (png_sig_cmp(header, 0, 8))
    {
        PyErr_SetString(
            PyExc_ValueError,
            "File not recognized as a PNG file");
        goto exit;
    }

    /* initialize stuff */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "png_create_read_struct failed");
        goto exit;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "png_create_info_struct failed");
        goto exit;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        PyErr_SetString(
            PyExc_RuntimeError, "Error building image");
        goto exit;
    }

    if (fp)
    {
        png_init_io(png_ptr, fp);
    }
    else
    {
        /* Read from a Python function */
        png_set_read_fn(png_ptr, (void*)py_fileobj, &read_png_data);
    }
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    width = info_ptr->width;
    height = info_ptr->height;
    bit_depth = info_ptr->bit_depth;

    // Unpack 1, 2, and 4-bit images
    if (bit_depth < 8)
    {
        png_set_packing(png_ptr);
    }

    // If sig bits are set, shift data
    png_color_8p sig_bit;
    if ((info_ptr->color_type != PNG_COLOR_TYPE_PALETTE) &&
        png_get_sBIT(png_ptr, info_ptr, &sig_bit))
    {
        png_set_shift(png_ptr, sig_bit);
    }

    // Convert big endian to little
    if (bit_depth == 16)
    {
        png_set_swap(png_ptr);
    }

    // Convert palletes to full RGB
    if (info_ptr->color_type == PNG_COLOR_TYPE_PALETTE)
    {
        png_set_palette_to_rgb(png_ptr);
    }

    // If there's an alpha channel convert gray to RGB
    if (info_ptr->color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    {
        png_set_gray_to_rgb(png_ptr);
    }

    png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    /* read file */
    row_pointers = new png_bytep[height];
    for (row = 0; row < height; row++)
    {
        row_pointers[row] = NULL;
    }

    for (row = 0; row < height; row++)
    {
        row_pointers[row] = new png_byte[png_get_rowbytes(png_ptr,info_ptr)];
    }

    png_read_image(png_ptr, row_pointers);

    dimensions[0] = height;  //numrows
    dimensions[1] = width;   //numcols
    if (info_ptr->color_type & PNG_COLOR_MASK_ALPHA)
    {
        dimensions[2] = 4;     //RGBA images
    }
    else if (info_ptr->color_type & PNG_COLOR_MASK_COLOR)
    {
        dimensions[2] = 3;     //RGB images
    }
    else
    {
        dimensions[2] = 1;     //Greyscale images
    }
    //For gray, return an x by y array, not an x by y by 1
    num_dims = (info_ptr->color_type & PNG_COLOR_MASK_COLOR) ? 3 : 2;

    if (result_type == NPY_FLOAT) {
        double max_value = (1 << ((bit_depth < 8) ? 8 : bit_depth)) - 1;

        A = (PyArrayObject *) PyArray_SimpleNew(num_dims, dimensions, NPY_FLOAT);

        if (A == NULL)
        {
            PyErr_SetString(
                PyExc_MemoryError,
                "Could not allocate image array");
            goto exit;
        }

        for (png_uint_32 y = 0; y < height; y++)
        {
            png_byte* row = row_pointers[y];
            for (png_uint_32 x = 0; x < width; x++)
            {
                size_t offset = y * A->strides[0] + x * A->strides[1];
                if (bit_depth == 16)
                {
                    png_uint_16* ptr = &reinterpret_cast<png_uint_16*>(row)[x * dimensions[2]];
                    for (png_uint_32 p = 0; p < (png_uint_32)dimensions[2]; p++)
                    {
                        *(float*)(A->data + offset + p*A->strides[2]) = (float)(ptr[p]) / max_value;
                    }
                }
                else
                {
                    png_byte* ptr = &(row[x * dimensions[2]]);
                    for (png_uint_32 p = 0; p < (png_uint_32)dimensions[2]; p++)
                    {
                        *(float*)(A->data + offset + p*A->strides[2]) = (float)(ptr[p]) / max_value;
                    }
                }
            }
        }
    } else if (result_type == NPY_UBYTE) {
        A = (PyArrayObject *) PyArray_SimpleNew(num_dims, dimensions, NPY_UBYTE);

        if (A == NULL)
        {
            PyErr_SetString(
                PyExc_MemoryError,
                "Could not allocate image array");
            goto exit;
        }

        for (png_uint_32 y = 0; y < height; y++)
        {
            png_byte* row = row_pointers[y];
            for (png_uint_32 x = 0; x < width; x++)
            {
                size_t offset = y * A->strides[0] + x * A->strides[1];
                if (bit_depth == 16)
                {
                    png_uint_16* ptr = &reinterpret_cast<png_uint_16*>(row)[x * dimensions[2]];
                    for (png_uint_32 p = 0; p < (png_uint_32)dimensions[2]; p++)
                    {
                        *(png_byte*)(A->data + offset + p*A->strides[2]) = ptr[p] >> 8;
                    }
                }
                else
                {
                    png_byte* ptr = &(row[x * dimensions[2]]);
                    for (png_uint_32 p = 0; p < (png_uint_32)dimensions[2]; p++)
                    {
                        *(png_byte*)(A->data + offset + p*A->strides[2]) = ptr[p];
                    }
                }
            }
        }
    }

    result = (PyObject *)A;

 exit:

    //free the png memory
    png_read_end(png_ptr, info_ptr);
#ifndef png_infopp_NULL
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
#else
    png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
#endif
    if (close_file)
    {
        fclose(fp);
    }
    for (row = 0; row < height; row++)
    {
        delete [] row_pointers[row];
    }
    delete [] row_pointers;

    return result;
}

extern "C" PyObject*
read_png_float(PyObject* py_file_obj)
{
    return _read_png(py_file_obj, NPY_FLOAT);
}

extern "C" PyObject*
read_png_uint8(PyObject* py_file_obj)
{
    return _read_png(py_file_obj, NPY_UBYTE);
}
