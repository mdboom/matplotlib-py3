/* -*- mode: c++; c-basic-offset: 4 -*- */

/* _backend_agg.h - A rewrite of _backend_agg using PyCXX to handle
   ref counting, etc..
*/

#ifndef __BACKEND_AGG_H
#define __BACKEND_AGG_H
#include <utility>

#include "agg_arrowhead.h"
#include "agg_basics.h"
#include "agg_bezier_arc.h"
#include "agg_color_rgba.h"
#include "agg_conv_concat.h"
#include "agg_conv_contour.h"
#include "agg_conv_curve.h"
#include "agg_conv_dash.h"
#include "agg_conv_marker.h"
#include "agg_conv_marker_adaptor.h"
#include "agg_math_stroke.h"
#include "agg_conv_stroke.h"
#include "agg_ellipse.h"
#include "agg_embedded_raster_fonts.h"
#include "agg_path_storage.h"
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"
#include "agg_pixfmt_gray.h"
#include "agg_alpha_mask_u8.h"
#include "agg_pixfmt_amask_adaptor.h"
#include "agg_rasterizer_outline.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_renderer_outline_aa.h"
#include "agg_renderer_raster_text.h"
#include "agg_renderer_scanline.h"
#include "agg_rendering_buffer.h"
#include "agg_scanline_bin.h"
#include "agg_scanline_u.h"
#include "agg_scanline_p.h"
#include "agg_vcgen_markers_term.h"

#include "agg_py_path_iterator.h"
#include "path_converters.h"

// These are copied directly from path.py, and must be kept in sync
#define STOP   0
#define MOVETO 1
#define LINETO 2
#define CURVE3 3
#define CURVE4 4
#define CLOSEPOLY 5

const size_t NUM_VERTICES[] = { 1, 1, 1, 2, 3, 1 };

typedef agg::pixfmt_rgba32 pixfmt;
typedef agg::renderer_base<pixfmt> renderer_base;
typedef agg::renderer_scanline_aa_solid<renderer_base> renderer_aa;
typedef agg::renderer_scanline_bin_solid<renderer_base> renderer_bin;
typedef agg::rasterizer_scanline_aa<agg::rasterizer_sl_clip_dbl> rasterizer;

typedef agg::scanline_p8 scanline_p8;
typedef agg::scanline_bin scanline_bin;
typedef agg::amask_no_clip_gray8 alpha_mask_type;

typedef agg::renderer_base<agg::pixfmt_gray8> renderer_base_alpha_mask_type;
typedef agg::renderer_scanline_aa_solid<renderer_base_alpha_mask_type> renderer_alpha_mask_type;

class BufferRegion
{
public:
    BufferRegion(const agg::rect_i &r, bool freemem = true) :
        rect(r), freemem(freemem)
    {
        width = r.x2 - r.x1;
        height = r.y2 - r.y1;
        stride = width * 4;
        data = new agg::int8u[stride * height];
    }

    virtual ~BufferRegion()
    {
        if (freemem)
        {
            delete [] data;
            data = NULL;
        }
    };

    agg::int8u* data;
    agg::rect_i rect;
    int width;
    int height;
    int stride;
    bool freemem;

    // set the x and y corners of the rectangle
    void set_x(const long x)
    {
        rect.x1 = x;
    }

    void set_y(const long y);
    {
        rect.y1 = y;
    }
};

template<class PathIterator>
class GCAgg
{
public:
    GCAgg() :
        dpi(150),
        alpha(1.0),
        color(1.0, 1.0, 1.0, 1.0),
        linewidth(1.0),
        isaa(true),
        cap(agg::butt_cap),
        join(agg::miter_join_revert),
        has_cliprect(false),
        has_clippath(false),
        snap_mode(SNAP_FALSE),
        has_hatchpath(false)
    {}

    double dpi;
    double alpha;
    agg::rgba color;
    double linewidth;

    bool isaa;
    agg::line_cap_e cap;
    agg::line_join_e join;

    bool has_cliprect;
    agg::rect_d cliprect;

    bool has_clippath;
    PathIterator clippath;
    agg::trans_affine clippath_trans;

    typedef std::vector<std::pair<double, double> > dash_t;
    double dashOffset;
    dash_t dashes;

    e_snap_mode snap_mode;

    bool has_hatchpath
    PathIterator hatchpath;

    double points_to_pixels(double points);
};

// the renderer
class RendererAgg
{
public:
    typedef std::pair<bool, agg::rgba> facepair_t;

    RendererAgg(unsigned int width, unsigned int height, double dpi, int debug);

    unsigned int get_width()
    {
        return width;
    }

    unsigned int get_height()
    {
        return height;
    }

    // the drawing methods
    template<class PathIterator>
    void draw_markers(
        const GCAgg &gc,
        PathIterator &marker_path,
        const agg::trans_affine &marker_trans,
        PathIterator &path,
        const agg::trans_affine &trans,
        const facepair_t &face);
    void draw_text_image(
        const GCAgg& gc,
        const unsigned char* buffer,
        int width, int height,
        int x, int y, double angle);
    void draw_image(
        const GCAgg& gc,
        const Image* image,
        double x, double y, double w, double h,
        bool has_affine, const agg::trans_affine& affine_trans);
    template<class PathIterator>
    void draw_path(
        const GCAgg& gc,
        PathIterator& path,
        const agg::trans_affine& trans,
        const facepair_t& face);
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

    BufferRegion* copy_from_bbox(const agg::rect_d& rect);
    void restore_region(const BufferRegion* const region);
    void restore_region2(
        const BufferRegion* const region,
        int x, int y, int xx1, int yy1, int xx2, int yy2);

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

    void* lastclippath;
    agg::trans_affine lastclippath_transform;

    static const size_t HATCH_SIZE = 72;
    agg::int8u hatchBuffer[HATCH_SIZE * HATCH_SIZE * 4];
    agg::rendering_buffer hatchRenderingBuffer;

    const int debug;

protected:
    double points_to_pixels(const Py::Object& points);
    agg::rgba rgb_to_color(const Py::SeqBase<Py::Object>& rgb, double alpha);

    template<class R>
    void set_clipbox(const Py::Object& cliprect, R& rasterizer);

    template<class PathIterator>
    void render_clippath(
        bool has_clippath,
        const PathIterator& clippath,
        const agg::trans_affine& clippath_trans);

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

// the extension module
class _backend_agg_module : public Py::ExtensionModule<_backend_agg_module>
{
public:
    _backend_agg_module()
        : Py::ExtensionModule<_backend_agg_module>("_backend_agg")
    {
        RendererAgg::init_type();
        BufferRegion::init_type();

        add_keyword_method("RendererAgg", &_backend_agg_module::new_renderer,
                           "RendererAgg(width, height, dpi)");
        initialize("The agg rendering backend");
    }

    virtual ~_backend_agg_module() {}

private:

    Py::Object new_renderer(const Py::Tuple &args, const Py::Dict &kws);
};

template<class PathIterator>
void
RendererAgg::render_clippath(
    bool has_clippath,
    const PathIterator& clippath,
    const agg::trans_affine& clippath_trans)
{
    typedef agg::conv_transform<PathIterator> transformed_path_t;
    typedef agg::conv_curve<transformed_path_t> curve_t;

    if (has_clippath &&
        (clippath.id() != lastclippath ||
         clippath_trans != lastclippath_transform))
    {
        create_alpha_buffers();
        agg::trans_affine trans(clippath_trans);
        trans *= agg::trans_affine_scaling(1.0, -1.0);
        trans *= agg::trans_affine_translation(0.0, (double)height);

        PathIterator clippath_iter(clippath);
        rendererBaseAlphaMask.clear(agg::gray8(0, 0));
        transformed_path_t transformed_clippath(clippath_iter, trans);
        agg::conv_curve<transformed_path_t> curved_clippath(transformed_clippath);
        theRasterizer.add_path(curved_clippath);
        rendererAlphaMask.color(agg::gray8(255, 255));
        agg::render_scanlines(theRasterizer, scanlineAlphaMask, rendererAlphaMask);
        lastclippath = clippath.id();
        lastclippath_transform = clippath_trans;
    }
}

#define MARKER_CACHE_SIZE 512

template<class PathIterator>
void
RendererAgg::draw_markers(
    const GCAgg &gc,
    PathIterator &marker_path,
    const agg::trans_affine &marker_trans,
    PathIterator &path,
    const agg::trans_affine &trans,
    const facepair_t &face)
{
    typedef agg::conv_transform<PathIterator>                  transformed_path_t;
    typedef PathSnapper<transformed_path_t>                    snap_t;
    typedef agg::conv_curve<snap_t>                            curve_t;
    typedef agg::conv_stroke<curve_t>                          stroke_t;
    typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type> pixfmt_amask_type;
    typedef agg::renderer_base<pixfmt_amask_type>              amask_ren_type;
    typedef agg::renderer_scanline_aa_solid<amask_ren_type>    amask_aa_renderer_type;
    typedef agg::renderer_scanline_bin_solid<amask_ren_type>   amask_bin_renderer_type;

    // Deal with the difference in y-axis direction
    marker_trans *= agg::trans_affine_scaling(1.0, -1.0);
    trans *= agg::trans_affine_scaling(1.0, -1.0);
    trans *= agg::trans_affine_translation(0.0, (double)height);

    transformed_path_t marker_path_transformed(marker_path, marker_trans);
    snap_t             marker_path_snapped(marker_path_transformed,
                                           gc.snap_mode,
                                           marker_path.total_vertices(),
                                           gc.linewidth);
    curve_t            marker_path_curve(marker_path_snapped);

    transformed_path_t path_transformed(path, trans);
    snap_t             path_snapped(path_transformed,
                                    gc.snap_mode,
                                    path.total_vertices(),
                                    1.0);
    curve_t            path_curve(path_snapped);
    path_curve.rewind(0);

    //maxim's suggestions for cached scanlines
    agg::scanline_storage_aa8 scanlines;
    theRasterizer.reset();
    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);

    agg::int8u  staticFillCache[MARKER_CACHE_SIZE];
    agg::int8u  staticStrokeCache[MARKER_CACHE_SIZE];
    agg::int8u* fillCache = staticFillCache;
    agg::int8u* strokeCache = staticStrokeCache;

    try
    {
        unsigned fillSize = 0;
        if (face.first)
        {
            theRasterizer.add_path(marker_path_curve);
            agg::render_scanlines(theRasterizer, slineP8, scanlines);
            fillSize = scanlines.byte_size();
            if (fillSize >= MARKER_CACHE_SIZE)
            {
                fillCache = new agg::int8u[fillSize];
            }
            scanlines.serialize(fillCache);
        }

        stroke_t stroke(marker_path_curve);
        stroke.width(gc.linewidth);
        stroke.line_cap(gc.cap);
        stroke.line_join(gc.join);
        theRasterizer.reset();
        theRasterizer.add_path(stroke);
        agg::render_scanlines(theRasterizer, slineP8, scanlines);
        unsigned strokeSize = scanlines.byte_size();
        if (strokeSize >= MARKER_CACHE_SIZE)
        {
            strokeCache = new agg::int8u[strokeSize];
        }
        scanlines.serialize(strokeCache);

        theRasterizer.reset_clipping();
        rendererBase.reset_clipping(true);
        set_clipbox(gc.cliprect, rendererBase);
        bool has_clippath = render_clippath(gc.clippath, gc.clippath_trans);

        double x, y;

        agg::serialized_scanlines_adaptor_aa8 sa;
        agg::serialized_scanlines_adaptor_aa8::embedded_scanline sl;

        agg::rect_d clipping_rect(
            -(scanlines.min_x() + 1.0),
            -(scanlines.min_y() + 1.0),
            width + scanlines.max_x() + 1.0,
            height + scanlines.max_y() + 1.0);

        if (has_clippath)
        {
            while (path_curve.vertex(&x, &y) != agg::path_cmd_stop)
            {
                if (MPL_notisfinite64(x) || MPL_notisfinite64(y))
                {
                    continue;
                }

                x = (double)(int)x;
                y = (double)(int)y;

                // Cull points outside the boundary of the image.
                // Values that are too large may overflow and create
                // segfaults.
                // http://sourceforge.net/tracker/?func=detail&aid=2865490&group_id=80706&atid=560720
                if (!clipping_rect.hit_test(x, y))
                {
                    continue;
                }

                pixfmt_amask_type pfa(pixFmt, alphaMask);
                amask_ren_type r(pfa);
                amask_aa_renderer_type ren(r);

                if (face.first)
                {
                    ren.color(face.second);
                    sa.init(fillCache, fillSize, x, y);
                    agg::render_scanlines(sa, sl, ren);
                }
                ren.color(gc.color);
                sa.init(strokeCache, strokeSize, x, y);
                agg::render_scanlines(sa, sl, ren);
            }
        }
        else
        {
            while (path_curve.vertex(&x, &y) != agg::path_cmd_stop)
            {
                if (MPL_notisfinite64(x) || MPL_notisfinite64(y))
                {
                    continue;
                }

                x = (double)(int)x;
                y = (double)(int)y;

                // Cull points outside the boundary of the image.
                // Values that are too large may overflow and create
                // segfaults.
                // http://sourceforge.net/tracker/?func=detail&aid=2865490&group_id=80706&atid=560720
                if (!clipping_rect.hit_test(x, y))
                {
                    continue;
                }

                if (face.first)
                {
                    rendererAA.color(face.second);
                    sa.init(fillCache, fillSize, x, y);
                    agg::render_scanlines(sa, sl, rendererAA);
                }

                rendererAA.color(gc.color);
                sa.init(strokeCache, strokeSize, x, y);
                agg::render_scanlines(sa, sl, rendererAA);
            }
        }
    }
    catch (...)
    {
        if (fillCache != staticFillCache)
            delete[] fillCache;
        if (strokeCache != staticStrokeCache)
            delete[] strokeCache;
        theRasterizer.reset_clipping();
        rendererBase.reset_clipping(true);
        throw;
    }

    if (fillCache != staticFillCache)
        delete[] fillCache;
    if (strokeCache != staticStrokeCache)
        delete[] strokeCache;

    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);

    return Py::Object();
}

template<class PathIterator>
void
RendererAgg::draw_path(
    const GCAgg& gc,
    PathIterator& path,
    const agg::trans_affine& trans,
    const facepair_t& face)
{
    typedef agg::conv_transform<PathIterator>  transformed_path_t;
    typedef PathNanRemover<transformed_path_t> nan_removed_t;
    typedef PathClipper<nan_removed_t>         clipped_t;
    typedef PathSnapper<clipped_t>             snapped_t;
    typedef PathSimplifier<snapped_t>          simplify_t;
    typedef agg::conv_curve<simplify_t>        curve_t;

    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(gc.cliprect, theRasterizer);
    bool has_clippath = render_clippath(gc.clippath, gc.clippath_trans);

    trans *= agg::trans_affine_scaling(1.0, -1.0);
    trans *= agg::trans_affine_translation(0.0, (double)height);
    bool clip = !face.first && gc.hatchpath.isNone() && !path.has_curves();
    bool simplify = path.should_simplify() && clip;

    transformed_path_t tpath(path, trans);
    nan_removed_t      nan_removed(tpath, true, path.has_curves());
    clipped_t          clipped(nan_removed, clip, width, height);
    snapped_t          snapped(clipped, gc.snap_mode, path.total_vertices(), gc.linewidth);
    simplify_t         simplified(snapped, simplify, path.simplify_threshold());
    curve_t            curve(simplified);

    _draw_path(curve, has_clippath, face, gc);
}

#endif

