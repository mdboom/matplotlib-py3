/* -*- mode: c++; c-basic-offset: 4 -*- */

#include "CXX/Extensions.hxx"
#include "backend_agg.h"
#include "agg_py_transforms.h"
#include "agg_py_path_iterator.h"

/**********************************************************************
 BufferRegion
 **********************************************************************/

class PyBufferRegion :
    public Py::PythonClass<PyBufferRegion>
{
public:
    PyBufferRegion(Py::PythonClassInstance *self, Py::Tuple &args, Py::Dict &kwds)
        : Py::PythonClass<PyBufferRegion>::PythonClass(self, args, kwds)
    {}

    static Py::PythonClassObject<PyBufferRegion>
    factory(BufferRegion* buf);

    virtual ~PyBufferRegion()
    {
        delete buf;
    }

    BufferRegion *buf;

    // static Py::PythonClassObject<BufferRegion> factory(
    //     const agg::rect_i &r, bool freemem = true);

    Py::Object set_x(const Py::Tuple &args);
    Py::Object set_y(const Py::Tuple &args);
    Py::Object get_extents();
    Py::Object to_string();
    Py::Object to_string_argb();

    static void init_type(void);
};

Py::PythonClassObject<PyBufferRegion>
PyBufferRegion::factory(BufferRegion* buf)
{
    Py::PythonClassObject<PyBufferRegion> o = Py::PythonClassObject<PyBufferRegion>(
        class_type.apply(Py::Tuple(0), Py::Dict()));
    o->getCxxObject()->buf = buf;
    return o;
}

Py::Object
PyBufferRegion::set_x(const Py::Tuple &args)
{
    args.verify_length(1);
    long x = Py::Int(args[0]);
    buf->set_x(x);
    return Py::Object();
}
PYCXX_VARARGS_METHOD_DECL(PyBufferRegion, set_x)

Py::Object
PyBufferRegion::set_y(const Py::Tuple &args)
{
    args.verify_length(1);
    long y = Py::Int(args[0]);
    buf->set_y(y);
    return Py::Object();
}
PYCXX_VARARGS_METHOD_DECL(PyBufferRegion, set_y)

Py::Object
PyBufferRegion::get_extents()
{
    args.verify_length(0);

    Py::Tuple extents(4);
    extents[0] = Py::Int(buf->rect.x1);
    extents[1] = Py::Int(buf->rect.y1);
    extents[2] = Py::Int(buf->rect.x2);
    extents[3] = Py::Int(buf->rect.y2);

    return extents;
}
PYCXX_NOARGS_METHOD_DECL(PyBufferRegion, get_extents)

Py::Object
PyBufferRegion::to_string()
{
    // owned=true to prevent memory leak
    #if PY3K
    return Py::Bytes(PyBytes_FromStringAndSize(
                         (const char*)buf->data, buf->height*buf->stride), true);
    #else
    return Py::String(PyString_FromStringAndSize(
                          (const char*)buf->data, buf->height*buf->stride), true);
    #endif
}
PYCXX_NOARGS_METHOD_DECL(PyBufferRegion, to_string)

Py::Object
PyBufferRegion::to_string_argb(const Py::Tuple &args)
{
    // owned=true to prevent memory leak
    Py_ssize_t length;
    unsigned char* pix;
    unsigned char* begin;
    unsigned char* end;
    unsigned char tmp;
    size_t i, j;

    #if PY3K
    PyObject* str = PyBytes_FromStringAndSize(
        (const char*)buf->data, buf->height * buf->stride);
    if (PyBytes_AsStringAndSize(str, (char**)&begin, &length))
    {
        throw Py::TypeError("Could not create memory for blit");
    }
    #else
    PyObject* str = PyString_FromStringAndSize(
        (const char*)buf->data, buf->height * buf->stride);
    if (PyString_AsStringAndSize(str, (char**)&begin, &length))
    {
        throw Py::TypeError("Could not create memory for blit");
    }
    #endif

    pix = begin;
    end = begin + (height * stride);
    for (i = 0; i < (size_t)height; ++i)
    {
        pix = begin + i * stride;
        for (j = 0; j < (size_t)width; ++j)
        {
            // Convert rgba to argb
            tmp = pix[2];
            pix[2] = pix[0];
            pix[0] = tmp;
            pix += 4;
        }
    }

    return Py::String(str, true);
}
PYCXX_NOARGS_METHOD_DECL(PyBufferRegion, to_string_argb)

void
PyBufferRegion::init_type()
{
    behaviors().name("BufferRegion");
    behaviors().doc("A wrapper to pass agg buffer objects to and from the python level");

    PYCXX_ADD_VARARGS_METHOD("set_x", &PyBufferRegion::set_x, "set_x(x)");
    PYCXX_ADD_VARARGS_METHOD("set_y", &PyBufferRegion::set_y, "set_y(y)");
    PYCXX_ADD_NOARGS_METHOD("get_extents", &PyBufferRegion::get_extents, "get_extents()");
    PYCXX_ADD_NOARGS_METHOD("to_string", &PyBufferRegion::to_string, "to_string()");
    PYCXX_ADD_NOARGS_METHOD("to_string_argb", &PyBufferRegion::to_string_argb, "to_string_argb()");

    behaviors().readyType();
}

/**********************************************************************
 GCAgg
 **********************************************************************/

/*
 Convert dashes from the Python representation as nested sequences to
 the C++ representation as a std::vector<std::pair<double, double> >
 (GCAgg::dash_t) */
static void
convert_dashes(const Py::Tuple& dashes, double dpi,
               GCAgg::dash_t& dashes_out, double& dashOffset_out)
{
    if (dashes.length() != 2)
    {
        throw Py::ValueError(
            Printf("Dash descriptor must be a length 2 tuple; found %d",
                   dashes.length()).str()
        );
    }

    dashes_out.clear();
    dashOffset_out = 0.0;
    if (dashes[0].ptr() == Py_None)
    {
        return;
    }

    dashOffset_out = double(Py::Float(dashes[0])) * dpi / 72.0;

    Py::SeqBase<Py::Object> dashSeq = dashes[1];

    size_t Ndash = dashSeq.length();
    if (Ndash % 2 != 0)
    {
        throw Py::ValueError(
            Printf("Dash sequence must be an even length sequence; found %d", Ndash).str()
        );
    }

    dashes_out.clear();
    dashes_out.reserve(Ndash / 2);

    double val0, val1;
    for (size_t i = 0; i < Ndash; i += 2)
    {
        val0 = double(Py::Float(dashSeq[i])) * dpi / 72.0;
        val1 = double(Py::Float(dashSeq[i+1])) * dpi / 72.0;
        dashes_out.push_back(std::make_pair(val0, val1));
    }
}

static void
fill_gc(GCAgg<PathIterator>& gc, const Py::Object &pygc, double dpi)
{
    /* TODO: Implement this in terms of getters, not private members */

    gc.dpi = dpi;

    gc.alpha = Py::Float(pygc.getAttr("_alpha"));

    Py::Tuple rgb = Py::Tuple(pygc.getAttr("_rgb"));
    gc.color = agg::rgba(Py::Float(rgb[0]),
                         Py::Float(rgb[1]),
                         Py::Float(rgb[2]),
                         gc.alpha);

    gc.linewidth = gc.points_to_pixels(pygc.getAttr("_linewidth"));

    gc.isaa = Py::Boolean(gc.getAttr("_antialiased"));

    std::string capstyle = Py::String(gc.getAttr("_capstyle"));
    if (capstyle == "butt")
    {
        gc.cap = agg::butt_cap;
    }
    else if (capstyle == "round")
    {
        gc.cap = agg::round_cap;
    }
    else if (capstyle == "projecting")
    {
        gc.cap = agg::square_cap;
    }
    else
    {
        throw Py::ValueError(
            Printf(
                "GC _capstyle attribute must be one of butt, round, projecting; found %s",
                capstyle.c_str()).str());
    }

    std::string joinstyle = Py::String(gc.getAttr("_joinstyle"));
    if (joinstyle == "miter")
    {
        gc.join = agg::miter_join_revert;
    }
    else if (joinstyle == "round")
    {
        gc.join = agg::round_join;
    }
    else if (joinstyle == "bevel")
    {
        gc.join = agg::bevel_join;
    }
    else
    {
        throw Py::ValueError(
            Printf(
                "GC _joinstyle attribute must be one of butt, round, projecting; found %s",
                joinstyle.c_str()).str());
    }

    Py::Object dash_obj(gc.getAttr("_dashes"));
    if (!dash_obj.isNone())
    {
        convert_dashes(dash_obj, dpi, gc.dashes, gc.dashOffset);
    }

    Py::Object cliprect(gc.getAttr("_cliprect"));
    gc.has_cliprect = py_convert_bbox(cliprect.ptr(), gc.cliprect);

    Py::Object method_obj = gc.getAttr("get_clip_path");
    Py::Callable method(method_obj);
    Py::Tuple path_and_transform = method.apply(Py::Tuple());
    if (path_and_transform[0].ptr() != Py_None)
    {
        gc.has_clippath = true;
        gc.clippath = PathIterator(path_and_transform[0]);
        gc.clippath_trans = py_to_agg_transformation_matrix(path_and_transform[1].ptr());
    } else {
        gc.has_clippath = false;
    }

    Py::Object method_obj = gc.getAttr("get_snap");
    Py::Callable method(method_obj);
    Py::Object snap = method.apply(Py::Tuple());
    if (snap.ptr() == NULL)
    {
        throw Py::Exception();
    }

    if (snap.isNone())
    {
        gc.snap_mode = SNAP_AUTO;
    }
    else if (snap.isTrue())
    {
        gc.snap_mode = SNAP_TRUE;
    }
    else
    {
        gc.snap_mode = SNAP_FALSE;
    }

    Py::Object method_obj = gc.getAttr("get_hatch_path");
    Py::Callable method(method_obj);
    hatchpath = method.apply(Py::Tuple());
    if (hatchpath.ptr() == NULL)
    {
        throw Py::Exception();
    }
    if (!hatchpath.isNone()) {
        gc.has_hatchpath = true;
        gc.hatchpath = PathIterator(hatchpath);
    } else {
        gc.has_hatchpath = false;
    }
}

/**********************************************************************
 RendererAgg
 **********************************************************************/

class PyRendererAgg :
    public Py::PythonClass<PyRendererAgg>
{
    typedef std::pair<bool, agg::rgba> facepair_t;
public:
    PyRendererAgg(Py::PythonClassInstance *self, Py::Tuple &args, Py::Dict &kwds);
    virtual ~PyRendererAgg()
    {
        delete ren;
    }

    RendererAgg *ren;

    static void init_type(void);

    // the drawing methods
    //Py::Object _draw_markers_nocache(const Py::Tuple & args);
    //Py::Object _draw_markers_cache(const Py::Tuple & args);
    Py::Object draw_markers(const Py::Tuple & args);
    Py::Object draw_text_image(const Py::Tuple & args);
    Py::Object draw_image(const Py::Tuple & args);
    Py::Object draw_path(const Py::Tuple & args);
    Py::Object draw_path_collection(const Py::Tuple & args);
    Py::Object draw_quad_mesh(const Py::Tuple& args);
    Py::Object draw_gouraud_triangle(const Py::Tuple& args);
    Py::Object draw_gouraud_triangles(const Py::Tuple& args);

    Py::Object write_rgba(const Py::Tuple & args);
    Py::Object tostring_rgb(const Py::Tuple & args);
    Py::Object tostring_argb(const Py::Tuple & args);
    Py::Object tostring_bgra(const Py::Tuple & args);
    Py::Object tostring_rgba_minimized(const Py::Tuple & args);
    Py::Object buffer_rgba(const Py::Tuple & args);
    Py::Object clear(const Py::Tuple & args);

    Py::Object copy_from_bbox(const Py::Tuple & args);
    Py::Object restore_region(const Py::Tuple & args);
    Py::Object restore_region2(const Py::Tuple & args);

    #if PY3K
    virtual int buffer_get( Py_buffer *, int flags );
    #endif

    virtual ~RendererAgg();

    static const size_t PIXELS_PER_INCH;
    unsigned int width, height;
    double dpi;
    size_t NUMBYTES;  //the number of bytes in buffer

    agg::int8u *pixBuffer;
    agg::rendering_buffer renderingBuffer;

    agg::int8u *alphaBuffer;
    agg::rendering_buffer alphaMaskRenderingBuffer;
    alpha_mask_type alphaMask;
    agg::pixfmt_gray8 pixfmtAlphaMask;
    renderer_base_alpha_mask_type rendererBaseAlphaMask;
    renderer_alpha_mask_type rendererAlphaMask;
    agg::scanline_p8 scanlineAlphaMask;

    scanline_p8 slineP8;
    scanline_bin slineBin;
    pixfmt pixFmt;
    renderer_base rendererBase;
    renderer_aa rendererAA;
    renderer_bin rendererBin;
    rasterizer theRasterizer;

    Py::Object lastclippath;
    agg::trans_affine lastclippath_transform;

    static const size_t HATCH_SIZE = 72;
    agg::int8u hatchBuffer[HATCH_SIZE * HATCH_SIZE * 4];
    agg::rendering_buffer hatchRenderingBuffer;

    const int debug;

protected:
    double points_to_pixels(const Py::Object& points);
    agg::rgba rgb_to_color(const Py::SeqBase<Py::Object>& rgb, double alpha);
    facepair_t _get_rgba_face(const Py::Object& rgbFace, double alpha);

    template<class R>
    void set_clipbox(const Py::Object& cliprect, R& rasterizer);

    bool render_clippath(const Py::Object& clippath, const agg::trans_affine& clippath_trans);

    template<class PathIteratorType>
    void _draw_path(PathIteratorType& path, bool has_clippath,
                    const facepair_t& face, const GCAgg& gc);

    template<class PathGenerator, int check_snap, int has_curves>
    Py::Object
    _draw_path_collection_generic
    (GCAgg&                         gc,
     agg::trans_affine              master_transform,
     const Py::Object&              cliprect,
     const Py::Object&              clippath,
     const agg::trans_affine&       clippath_trans,
     const PathGenerator&           path_generator,
     const Py::SeqBase<Py::Object>& transforms_obj,
     const Py::Object&              offsets_obj,
     const agg::trans_affine&       offset_trans,
     const Py::Object&              facecolors_obj,
     const Py::Object&              edgecolors_obj,
     const Py::SeqBase<Py::Float>&  linewidths,
     const Py::SeqBase<Py::Object>& linestyles_obj,
     const Py::SeqBase<Py::Int>&    antialiaseds);

    void
    _draw_gouraud_triangle(
        const double* points, const double* colors,
        agg::trans_affine trans, bool has_clippath);

private:
    void create_alpha_buffers();
};

static RendererAgg::facepair_t
get_rgba_face(const Py::Object& rgbFace, double alpha)
{
    _VERBOSE("RendererAgg::_get_rgba_face");
    std::pair<bool, agg::rgba> face;

    if (rgbFace.ptr() == Py_None)
    {
        face.first = false;
    }
    else
    {
        face.first = true;
        Py::Tuple rgb = Py::Tuple(rgbFace);
        face.second = rgb_to_color(rgb, alpha);
    }
    return face;
}



PyMODINIT_FUNC
#if PY3K
PyInit__backend_agg(void)
#else
init_backend_agg(void)
#endif
{
    _VERBOSE("init_backend_agg");

    import_array();

    static _backend_agg_module* _backend_agg = NULL;
    _backend_agg = new _backend_agg_module;

    #if PY3K
    return _backend_agg->module().ptr();
    #endif
}
