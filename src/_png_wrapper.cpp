/* -*- mode: c++; c-basic-offset: 4 -*- */

#include "CXX/Extensions.hxx"
#include "numpy/arrayobject.h"
#include "_png.h"
#include "mplutils.h"

#if PY_MAJOR_VERSION >= 3
#define PY3K 1
#else
#define PY3K 0
#endif

// the extension module
class _png_module : public Py::ExtensionModule<_png_module>
{
public:
    _png_module()
            : Py::ExtensionModule<_png_module>("_png")
    {
        add_varargs_method("write_png", &_png_module::write_png,
                           "write_png(buffer, width, height, fileobj, dpi=None)");
        add_varargs_method("read_png", &_png_module::read_png_float,
                           "read_png(fileobj)");
        add_varargs_method("read_png_float", &_png_module::read_png_float,
                           "read_png_float(fileobj)");
        add_varargs_method("read_png_uint8", &_png_module::read_png_uint8,
                           "read_png_uint8(fileobj)");
        initialize("Module to write PNG files");
    }

    virtual ~_png_module() {}

private:
    Py::Object write_png(const Py::Tuple& args);
    Py::Object read_png_uint8(const Py::Tuple& args);
    Py::Object read_png_float(const Py::Tuple& args);
    PyObject* _read_png(const Py::Object& py_fileobj, const int datatype);
};

static void
write_png_data(void* file_obj, unsigned char* data, size_t length)
{
    PyObject* py_file_obj = (PyObject*)file_obj;
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
flush_png_data(void* fileobj)
{
    PyObject* py_file_obj = (PyObject*)fileobj;
    PyObject* flush_method = PyObject_GetAttrString(py_file_obj, "flush");
    PyObject* result = NULL;
    if (flush_method)
    {
        result = PyObject_CallFunction(flush_method, (char *)"");
    }
    Py_XDECREF(flush_method);
    Py_XDECREF(result);
}

Py::Object _png_module::write_png(const Py::Tuple& args)
{
    args.verify_length(4, 5);

    FILE *fp = NULL;
    bool close_file = false;
    Py::Object buffer_obj = Py::Object(args[0]);
    PyObject* buffer = buffer_obj.ptr();
    double dpi = -1.0;
    if (!PyObject_CheckReadBuffer(buffer))
    {
        throw Py::TypeError("First argument must be an rgba buffer.");
    }

    const void* pixBuffer = NULL;
    ssize_t pixBufferLength = 0;
    if (PyObject_AsReadBuffer(buffer, &pixBuffer, &pixBufferLength))
    {
        throw Py::ValueError("Couldn't get data from read buffer.");
    }
    int width = (int)Py::Int(args[1]);
    int height = (int)Py::Int(args[2]);

    if (pixBufferLength < width * height * 4)
    {
        throw Py::ValueError("Buffer and width, height don't seem to match.");
    }

    Py::Object py_fileobj = Py::Object(args[3]);
#if PY3K
    int fd = PyObject_AsFileDescriptor(py_fileobj.ptr());
    PyErr_Clear();
#endif
    if (py_fileobj.isString())
    {
        std::string fileName = Py::String(py_fileobj);
        const char *file_name = fileName.c_str();
        if ((fp = fopen(file_name, "wb")) == NULL)
        {
            throw Py::RuntimeError(
                Printf("Could not open file %s", file_name).str());
        }
        close_file = true;
    }
#if PY3K
    else if (fd != -1)
    {
        fp = fdopen(fd, "w");
    }
#else
    else if (PyFile_CheckExact(py_fileobj.ptr()))
    {
        fp = PyFile_AsFile(py_fileobj.ptr());
    }
#endif
    else
    {
        PyObject* write_method = PyObject_GetAttrString(
            py_fileobj.ptr(), "write");
        if (!(write_method && PyCallable_Check(write_method)))
        {
            Py_XDECREF(write_method);
            throw Py::TypeError(
                "Object does not appear to be a 8-bit string path or a Python file-like object");
        }
        Py_XDECREF(write_method);
    }

    if (args.size() == 5)
    {
        dpi = Py::Float(args[4]);
    }

    try
    {
        ::write_png(pixBuffer, width, height, fp,
                    py_fileobj.ptr(), &write_png_data, &flush_png_data, dpi);
    }
    catch (const char* e)
    {
        if (fp && close_file)
        {
            fclose(fp);
        }
        throw Py::RuntimeError(e);
    }

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

    return Py::Object();
}

static void
read_png_data(void* file_obj, unsigned char* data, size_t length)
{
    PyObject* py_file_obj = (PyObject*)file_obj;
    PyObject* read_method = PyObject_GetAttrString(py_file_obj, "read");
    PyObject* result = NULL;
    char *buffer;
    Py_ssize_t bufflen;
    if (read_method)
    {
        result = PyObject_CallFunction(read_method, (char *)"i", length);
    }

    // TODO: Is is possible to avoid this extra copy?
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

PyObject*
_png_module::_read_png(const Py::Object& py_fileobj, const int data_type)
{
    FILE* fp = NULL;
    bool close_file = false;
    readinfo info;

#if PY3K
    int fd = PyObject_AsFileDescriptor(py_fileobj.ptr());
    PyErr_Clear();
#endif

    if (py_fileobj.isString())
    {
        std::string fileName = Py::String(py_fileobj);
        const char *file_name = fileName.c_str();
        if ((fp = fopen(file_name, "rb")) == NULL)
        {
            throw Py::RuntimeError(
                Printf("Could not open file %s for reading", file_name).str());
        }
        close_file = true;
    }
#if PY3K
    else if (fd != -1) {
        fp = fdopen(fd, "r");
    }
#else
    else if (PyFile_CheckExact(py_fileobj.ptr()))
    {
        fp = PyFile_AsFile(py_fileobj.ptr());
    }
#endif
    else
    {
        PyObject* read_method = PyObject_GetAttrString(py_fileobj.ptr(), "read");
        if (!(read_method && PyCallable_Check(read_method)))
        {
            Py_XDECREF(read_method);
            throw Py::TypeError("Object does not appear to be a 8-bit string path or a Python file-like object");
        }
        Py_XDECREF(read_method);
    }

    // TODO: When planes == 1, numdims should == 2

    try {
        ::read_png_header(fp, py_fileobj.ptr(), &read_png_data, &info);
    } catch (const char* e) {
        if (close_file)
        {
            fclose(fp);
        }
        throw Py::RuntimeError(e);
    }

    npy_intp dimensions[3];
    dimensions[0] = info.height;
    dimensions[1] = info.width;
    dimensions[2] = info.planes;
    int num_dims = (info.planes == 1) ? 2 : 3;
    PyArrayObject* A = (PyArrayObject*)PyArray_SimpleNew(
        num_dims, dimensions, data_type);

    if (A == NULL) {
        if (close_file)
        {
            fclose(fp);
        }
        throw Py::MemoryError("Could not allocate array");
    }
    int png_data_type = (data_type == NPY_FLOAT) ? -32 : 8;

    try {
        ::read_png_content(fp, py_fileobj.ptr(), &read_png_data,
                           PyArray_DATA(A), png_data_type, &info);
    } catch (const char* e) {
        if (close_file)
        {
            fclose(fp);
        }
        Py_XDECREF(A);
        throw Py::RuntimeError(e);
    }

    if (close_file)
    {
        fclose(fp);
    }

    return (PyObject*)A;
}

Py::Object
_png_module::read_png_float(const Py::Tuple& args)
{
    args.verify_length(1);
    return Py::asObject(_read_png(args[0], NPY_FLOAT));
}

Py::Object
_png_module::read_png_uint8(const Py::Tuple& args)
{
    args.verify_length(1);
    return Py::asObject(_read_png(args[0], NPY_UINT8));
}

PyMODINIT_FUNC
#if PY3K
PyInit__png(void)
#else
init_png(void)
#endif
{
    import_array();

    static _png_module* _png = NULL;
    _png = new _png_module;

#if PY3K
    return _png->module().ptr();
#endif
}
