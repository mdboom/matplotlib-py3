/* -*- mode: c++; c-basic-offset: 4 -*- */

/* A rewrite of _backend_agg using PyCXX to handle ref counting, etc..
 */

/* Python API mandates Python.h is included *first* */
#include "Python.h"

#include "ft2font.h"
#include "_image.h"
#include "_backend_agg.h"
#include "mplutils.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <time.h>
#include <algorithm>

#include "agg_conv_curve.h"
#include "agg_conv_transform.h"
#include "agg_image_accessors.h"
#include "agg_renderer_primitives.h"
#include "agg_scanline_storage_aa.h"
#include "agg_scanline_storage_bin.h"
#include "agg_span_allocator.h"
#include "agg_span_image_filter_gray.h"
#include "agg_span_image_filter_rgba.h"
#include "agg_span_interpolator_linear.h"
#include "agg_span_pattern_rgba.h"
#include "agg_span_gouraud_rgba.h"
#include "agg_conv_shorten_path.h"
#include "util/agg_color_conv_rgb8.h"

#include "MPL_isnan.h"

#include "numpy/arrayobject.h"
#include "agg_py_transforms.h"

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif
#ifndef M_PI_4
#define M_PI_4     0.785398163397448309616
#endif
#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923
#endif

/**********************************************************************
 RendererAgg
 **********************************************************************/

RendererAgg::RendererAgg(unsigned int width, unsigned int height, double dpi,
                         int debug) :
    width(width),
    height(height),
    dpi(dpi),
    NUMBYTES(width*height*4),
    pixBuffer(NULL),
    renderingBuffer(),
    alphaBuffer(NULL),
    alphaMaskRenderingBuffer(),
    alphaMask(alphaMaskRenderingBuffer),
    pixfmtAlphaMask(alphaMaskRenderingBuffer),
    rendererBaseAlphaMask(),
    rendererAlphaMask(),
    scanlineAlphaMask(),
    slineP8(),
    slineBin(),
    pixFmt(),
    rendererBase(),
    rendererAA(),
    rendererBin(),
    theRasterizer(),
    debug(debug)
{
    _VERBOSE("RendererAgg::RendererAgg");
    unsigned stride(width*4);

    pixBuffer = new agg::int8u[NUMBYTES];
    renderingBuffer.attach(pixBuffer, width, height, stride);
    pixFmt.attach(renderingBuffer);
    rendererBase.attach(pixFmt);
    rendererBase.clear(agg::rgba(1, 1, 1, 0));
    rendererAA.attach(rendererBase);
    rendererBin.attach(rendererBase);
    hatchRenderingBuffer.attach(hatchBuffer, HATCH_SIZE, HATCH_SIZE,
                                HATCH_SIZE*4);
}

RendererAgg::~RendererAgg()
{
    _VERBOSE("RendererAgg::~RendererAgg");

    delete [] alphaBuffer;
    delete [] pixBuffer;
}

void
RendererAgg::create_alpha_buffers()
{
    if (!alphaBuffer)
    {
        unsigned stride(width*4);
        alphaBuffer = new agg::int8u[NUMBYTES];
        alphaMaskRenderingBuffer.attach(alphaBuffer, width, height, stride);
        rendererBaseAlphaMask.attach(pixfmtAlphaMask);
        rendererAlphaMask.attach(rendererBaseAlphaMask);
    }
}

template<class R>
void
RendererAgg::set_clipbox(bool has_cliprect, const agg::rect_d& cliprect, R& rasterizer)
{
    if (has_cliprect)
    {
        rasterizer.clip_box(int(mpl_round(cliprect.x1)),
                            height - int(mpl_round(cliprect.y1)),
                            int(mpl_round(cliprect.x2)),
                            height - int(mpl_round(cliprect.y2)));
    }
    else
    {
        rasterizer.clip_box(0, 0, width, height);
    }
}

BufferRegion*
RendererAgg::copy_from_bbox(const agg::rect_d &in_rect)
{
    agg::rect_i rect((int)in_rect.x1, height - (int)in_rect.y1,
                     (int)in_rect.x2, height - (int)in_rect.y2);

    BufferRegion* reg = NULL;
    reg = new BufferRegion(rect, true);

    try
    {
        agg::rendering_buffer rbuf;
        rbuf.attach(reg->data, reg->width, reg->height, reg->stride);

        pixfmt pf(rbuf);
        renderer_base rb(pf);
        rb.copy_from(renderingBuffer, &rect, -rect.x1, -rect.y1);
    }
    catch (...)
    {
        delete reg;
        throw "An unknown error occurred in copy_from_bbox";
    }

    return reg;
}

void
RendererAgg::restore_region(const BufferRegion* const region)
{
    if (region->data == NULL)
    {
        throw "Cannot restore_region from NULL data";
    }

    agg::rendering_buffer rbuf;
    rbuf.attach(region->data,
                region->width,
                region->height,
                region->stride);

    rendererBase.copy_from(rbuf, 0, region->rect.x1, region->rect.y1);
}

// Restore the part of the saved region with offsets
void
RendererAgg::restore_region2(
    const BufferRegion* const region,
    int x, int y, int xx1, int yy1, int xx2, int yy2)
{
    if (region->data == NULL)
    {
        throw "Cannot restore_region from NULL data";
    }

    agg::rect_i rect(xx1 - region->rect.x1, (yy1 - region->rect.y1),
                     xx2 - region->rect.x1, (yy2 - region->rect.y1));

    agg::rendering_buffer rbuf;
    rbuf.attach(region->data,
                region->width,
                region->height,
                region->stride);

    rendererBase.copy_from(rbuf, &rect, x, y);
}

/**
 * This is a custom span generator that converts spans in the
 * 8-bit inverted greyscale font buffer to rgba that agg can use.
 */
template<class ChildGenerator>
class font_to_rgba
{
public:
    typedef ChildGenerator child_type;
    typedef agg::rgba8 color_type;
    typedef typename child_type::color_type child_color_type;
    typedef agg::span_allocator<child_color_type> span_alloc_type;

private:
    child_type* _gen;
    color_type _color;
    span_alloc_type _allocator;

public:
    font_to_rgba(child_type* gen, color_type color) :
        _gen(gen),
        _color(color)
    {

    }

    inline void
    generate(color_type* output_span, int x, int y, unsigned len)
    {
        _allocator.allocate(len);
        child_color_type* input_span = _allocator.span();
        _gen->generate(input_span, x, y, len);

        do
        {
            *output_span = _color;
            output_span->a = ((unsigned int)_color.a *
                              (unsigned int)input_span->v) >> 8;
            ++output_span;
            ++input_span;
        }
        while (--len);
    }

    void
    prepare()
    {
        _gen->prepare();
    }
};

// MGDTODO: Support clip paths
void
RendererAgg::draw_text_image(
    const GCAgg& gc,
    const unsigned char* buffer,
    int width, int height,
    int x, int y, double angle)
{
    typedef agg::span_allocator<agg::gray8> gray_span_alloc_type;
    typedef agg::span_allocator<agg::rgba8> color_span_alloc_type;
    typedef agg::span_interpolator_linear<> interpolator_type;
    typedef agg::image_accessor_clip<agg::pixfmt_gray8> image_accessor_type;
    typedef agg::span_image_filter_gray<image_accessor_type,
                                        interpolator_type> image_span_gen_type;
    typedef font_to_rgba<image_span_gen_type> span_gen_type;
    typedef agg::renderer_scanline_aa<renderer_base, color_span_alloc_type,
                                      span_gen_type> renderer_type;

    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(gc.cliprect, theRasterizer);

    agg::rendering_buffer srcbuf((agg::int8u*)buffer, width, height, width);
    agg::pixfmt_gray8 pixf_img(srcbuf);

    agg::trans_affine mtx;
    mtx *= agg::trans_affine_translation(0, -height);
    mtx *= agg::trans_affine_rotation(-angle * agg::pi / 180.0);
    mtx *= agg::trans_affine_translation(x, y);

    agg::path_storage rect;
    rect.move_to(0, 0);
    rect.line_to(width, 0);
    rect.line_to(width, height);
    rect.line_to(0, height);
    rect.line_to(0, 0);
    agg::conv_transform<agg::path_storage> rect2(rect, mtx);

    agg::trans_affine inv_mtx(mtx);
    inv_mtx.invert();

    agg::image_filter_lut filter;
    filter.calculate(agg::image_filter_spline36());
    interpolator_type interpolator(inv_mtx);
    color_span_alloc_type sa;
    image_accessor_type ia(pixf_img, 0);
    image_span_gen_type image_span_generator(ia, interpolator, filter);
    span_gen_type output_span_generator(&image_span_generator, gc.color);
    renderer_type ri(rendererBase, sa, output_span_generator);

    theRasterizer.add_path(rect2);
    agg::render_scanlines(theRasterizer, slineP8, ri);
}

void
RendererAgg::draw_image(
    const GCAgg& gc,
    const Image* image,
    double x, double y, double w, double h,
    bool has_affine, const agg::trans_affine& affine_trans)
{
    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(gc.cliprect, theRasterizer);
    has_clippath = render_clippath(gc.clippath, gc.clippath_trans);

    image->flipud_out();
    pixfmt pixf(*(image->rbufOut));

    if (has_affine | has_clippath)
    {
        agg::trans_affine mtx;
        agg::path_storage rect;

        if (has_affine)
        {
            mtx *= agg::trans_affine_scaling(1, -1);
            mtx *= agg::trans_affine_translation(0, image->rowsOut);
            mtx *= agg::trans_affine_scaling(w / (image->colsOut),
                                             h / (image->rowsOut));
            mtx *= agg::trans_affine_translation(x, y);
            mtx *= affine_trans;
            mtx *= agg::trans_affine_scaling(1.0, -1.0);
            mtx *= agg::trans_affine_translation(0.0, (double) height);
        }
        else
        {
            mtx *= agg::trans_affine_translation(
                (int)x,
                (int)(height - (y + image->rowsOut)));
        }

        rect.move_to(0, 0);
        rect.line_to(image->colsOut, 0);
        rect.line_to(image->colsOut, image->rowsOut);
        rect.line_to(0, image->rowsOut);
        rect.line_to(0, 0);

        agg::conv_transform<agg::path_storage> rect2(rect, mtx);

        agg::trans_affine inv_mtx(mtx);
        inv_mtx.invert();

        typedef agg::span_allocator<agg::rgba8> color_span_alloc_type;
        typedef agg::image_accessor_clip<agg::pixfmt_rgba32>
            image_accessor_type;
        typedef agg::span_interpolator_linear<> interpolator_type;
        typedef agg::span_image_filter_rgba_nn<image_accessor_type,
                                               interpolator_type> image_span_gen_type;

        color_span_alloc_type sa;
        image_accessor_type ia(pixf, agg::rgba8(0, 0, 0, 0));
        interpolator_type interpolator(inv_mtx);
        image_span_gen_type image_span_generator(ia, interpolator);

        if (has_clippath)
        {
            typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type>
                pixfmt_amask_type;
            typedef agg::renderer_base<pixfmt_amask_type> amask_ren_type;
            typedef agg::renderer_scanline_aa<amask_ren_type,
                                              color_span_alloc_type,
                                              image_span_gen_type>
                renderer_type_alpha;

            pixfmt_amask_type pfa(pixFmt, alphaMask);
            amask_ren_type r(pfa);
            renderer_type_alpha ri(r, sa, image_span_generator);

            theRasterizer.add_path(rect2);
            agg::render_scanlines(theRasterizer, slineP8, ri);
        }
        else
        {
            typedef agg::renderer_base<pixfmt> ren_type;
            typedef agg::renderer_scanline_aa<ren_type,
                                              color_span_alloc_type,
                                              image_span_gen_type>
                renderer_type;

            ren_type r(pixFmt);
            renderer_type ri(r, sa, image_span_generator);

            theRasterizer.add_path(rect2);
            agg::render_scanlines(theRasterizer, slineP8, ri);
        }

    }
    else
    {
        set_clipbox(gc.cliprect, rendererBase);
        rendererBase.blend_from(pixf, 0, (int)x, (int)(height - (y + image->rowsOut)));
    }

    rendererBase.reset_clipping(true);
    image->flipud_out();
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


template<class PathIterator,
         class PathGenerator,
         class OffsetsArray,
         class FaceColorsArray,
         class EdgeColorsArray,
         class LineWidthsArray,
         class LineStylesArray,
         class AntialiasedsArray,
         int check_snap,
         int has_curves>
void
RendererAgg::_draw_path_collection_generic(
    const GCAgg&                   gc,
    bool                           has_cliprect,
    const agg::rect_d&             cliprect,
    bool                           has_clippath,
    PathIterator&                  clippath,
    const agg::trans_affine&       clippath_trans,
    const PathGenerator&           path_generator,
    const transforms_vector_t&     transforms,
    const OffsetsArray&            offsets,
    const agg::trans_affine&       offset_trans,
    const FaceColorsArray&         facecolors,
    const EdgeColorsArray&         edgecolors,
    const LineWidthsArray&         linewidths,
    const LineStylesArray&         linestyles,
    const AntialiasedsArray&       antialiaseds)
{
    typedef agg::conv_transform<typename PathGenerator::path_iterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t>                         nan_removed_t;
    typedef PathClipper<nan_removed_t>                                 clipped_t;
    typedef PathSnapper<clipped_t>                                     snapped_t;
    typedef agg::conv_curve<snapped_t>                                 snapped_curve_t;
    typedef agg::conv_curve<clipped_t>                                 curve_t;

    size_t Npaths      = path_generator.num_paths();
    size_t Noffsets    = offsets.size();
    size_t N           = std::max(Npaths, Noffsets);
    size_t Ntransforms = std::min(transforms.size(), N);
    size_t Nfacecolors = facecolors.size();
    size_t Nedgecolors = edgecolors.size();
    size_t Nlinewidths = linewidths.size();
    size_t Nlinestyles = std::min(linestyles.size(), N);
    size_t Naa         = antialiaseds.size();

    if ((Nfacecolors == 0 && Nedgecolors == 0) || Npaths == 0)
    {
        return;
    }

    size_t i = 0;

    // Handle any clipping globally
    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(cliprect, theRasterizer);
    bool has_clippath = render_clippath(clippath, clippath_trans);

    // Set some defaults, assuming no face or edge
    gc.linewidth = 0.0;
    facepair_t face;
    face.first = Nfacecolors != 0;
    agg::trans_affine trans;

    for (i = 0; i < N; ++i)
    {
        typename PathGenerator::path_iterator path = path_generator(i);

        trans = transforms[i % Ntransforms];

        if (Noffsets)
        {
            double xo, yo;
            offsets.get(i % Noffsets, &xo, &yo);
            offset_trans.transform(&xo, &yo);
            trans *= agg::trans_affine_translation(xo, yo);
        }

        // These transformations must be done post-offsets
        trans *= agg::trans_affine_scaling(1.0, -1.0);
        trans *= agg::trans_affine_translation(0.0, (double)height);

        if (Nfacecolors)
        {
            facecolors.get(i % Nfacecolors, &face.second);
        }

        if (Nedgecolors)
        {
            edgecolors.get(i % Nedgecolors, &gc.color);

            if (Nlinewidths)
            {
                gc.linewidth = linewidths.get(i % Nlinewidths) * dpi / 72.0;
            }
            else
            {
                gc.linewidth = 1.0;
            }
            if (Nlinestyles)
            {
                linestyles.get(i % Nlinestyles, &gc.dashes, &gc.dashOffset);
            }
        }

        bool do_clip = !face.first && gc.hatchpath.isNone() && !has_curves;

        if (check_snap)
        {
            gc.isaa = Py::Boolean(antialiaseds[i % Naa]);

            transformed_path_t tpath(path, trans);
            nan_removed_t      nan_removed(tpath, true, has_curves);
            clipped_t          clipped(nan_removed, do_clip, width, height);
            snapped_t          snapped(clipped, gc.snap_mode,
                                       path.total_vertices(), gc.linewidth);
            if (has_curves)
            {
                snapped_curve_t curve(snapped);
                _draw_path(curve, has_clippath, face, gc);
            }
            else
            {
                _draw_path(snapped, has_clippath, face, gc);
            }
        }
        else
        {
            gc.isaa = Py::Boolean(antialiaseds[i % Naa]);

            transformed_path_t tpath(path, trans);
            nan_removed_t      nan_removed(tpath, true, has_curves);
            clipped_t          clipped(nan_removed, do_clip, width, height);
            if (has_curves)
            {
                curve_t curve(clipped);
                _draw_path(curve, has_clippath, face, gc);
            }
            else
            {
                _draw_path(clipped, has_clippath, face, gc);
            }
        }
    }
}

class PathListGenerator
{
    const Py::SeqBase<Py::Object>& m_paths;
    size_t m_npaths;

public:
    typedef PathIterator path_iterator;

    inline
    PathListGenerator(const Py::SeqBase<Py::Object>& paths) :
        m_paths(paths), m_npaths(paths.size())
    {

    }

    inline size_t
    num_paths() const
    {
        return m_npaths;
    }

    inline path_iterator
    operator()(size_t i) const
    {
        return PathIterator(m_paths[i % m_npaths]);
    }
};


Py::Object
RendererAgg::draw_path_collection(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::draw_path_collection");
    args.verify_length(12);

    Py::Object gc_obj = args[0];
    GCAgg gc(gc_obj, dpi);
    agg::trans_affine       master_transform = py_to_agg_transformation_matrix(args[1].ptr());
    Py::SeqBase<Py::Object> path   = args[2];
    PathListGenerator       path_generator(path);
    Py::SeqBase<Py::Object> transforms_obj   = args[3];
    Py::Object              offsets_obj      = args[4];
    agg::trans_affine       offset_trans     = py_to_agg_transformation_matrix(args[5].ptr());
    Py::Object              facecolors_obj   = args[6];
    Py::Object              edgecolors_obj   = args[7];
    Py::SeqBase<Py::Float>  linewidths       = args[8];
    Py::SeqBase<Py::Object> linestyles_obj   = args[9];
    Py::SeqBase<Py::Int>    antialiaseds     = args[10];
    // We don't actually care about urls for Agg, so just ignore it.
    // Py::SeqBase<Py::Object> urls             = args[11];

    try
    {
        _draw_path_collection_generic<PathListGenerator, 0, 1>
        (gc,
         master_transform,
         gc.cliprect,
         gc.clippath,
         gc.clippath_trans,
         path_generator,
         transforms_obj,
         offsets_obj,
         offset_trans,
         facecolors_obj,
         edgecolors_obj,
         linewidths,
         linestyles_obj,
         antialiaseds);
    }
    catch (const char *e)
    {
        throw Py::RuntimeError(e);
    }

    return Py::Object();
}


class QuadMeshGenerator
{
    size_t m_meshWidth;
    size_t m_meshHeight;
    PyArrayObject* m_coordinates;

    class QuadMeshPathIterator
    {
        size_t m_iterator;
        size_t m_m, m_n;
        PyArrayObject* m_coordinates;

    public:
        QuadMeshPathIterator(size_t m, size_t n, PyArrayObject* coordinates) :
            m_iterator(0), m_m(m), m_n(n), m_coordinates(coordinates)
        {

        }

    private:
        inline unsigned
        vertex(unsigned idx, double* x, double* y)
        {
            size_t m = m_m + ((idx     & 0x2) >> 1);
            size_t n = m_n + (((idx + 1) & 0x2) >> 1);
            double* pair = (double*)PyArray_GETPTR2(m_coordinates, n, m);
            *x = *pair++;
            *y = *pair;
            return (idx) ? agg::path_cmd_line_to : agg::path_cmd_move_to;
        }

    public:
        inline unsigned
        vertex(double* x, double* y)
        {
            if (m_iterator >= total_vertices())
            {
                return agg::path_cmd_stop;
            }
            return vertex(m_iterator++, x, y);
        }

        inline void
        rewind(unsigned path_id)
        {
            m_iterator = path_id;
        }

        inline unsigned
        total_vertices()
        {
            return 5;
        }

        inline bool
        should_simplify()
        {
            return false;
        }
    };

public:
    typedef QuadMeshPathIterator path_iterator;

    inline
    QuadMeshGenerator(size_t meshWidth, size_t meshHeight, PyObject* coordinates) :
        m_meshWidth(meshWidth), m_meshHeight(meshHeight), m_coordinates(NULL)
    {
        PyArrayObject* coordinates_array = \
            (PyArrayObject*)PyArray_ContiguousFromObject(
                coordinates, PyArray_DOUBLE, 3, 3);

        if (!coordinates_array)
        {
            throw Py::ValueError("Invalid coordinates array.");
        }

        m_coordinates = coordinates_array;
    }

    inline
    ~QuadMeshGenerator()
    {
        Py_XDECREF(m_coordinates);
    }

    inline size_t
    num_paths() const
    {
        return m_meshWidth * m_meshHeight;
    }

    inline path_iterator
    operator()(size_t i) const
    {
        return QuadMeshPathIterator(i % m_meshWidth, i / m_meshWidth, m_coordinates);
    }
};

Py::Object
RendererAgg::draw_quad_mesh(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::draw_quad_mesh");
    args.verify_length(10);

    //segments, trans, clipbox, colors, linewidths, antialiaseds
    GCAgg gc(args[0], dpi);
    agg::trans_affine master_transform = py_to_agg_transformation_matrix(args[1].ptr());
    size_t            mesh_width       = (long)Py::Int(args[2]);
    size_t            mesh_height      = (long)Py::Int(args[3]);
    Py::Object        coordinates      = args[4];
    Py::Object        offsets_obj      = args[5];
    agg::trans_affine offset_trans     = py_to_agg_transformation_matrix(args[6].ptr());
    Py::Object        facecolors_obj   = args[7];
    bool              antialiased      = (bool)Py::Boolean(args[8]);
    bool              showedges        = (bool)Py::Boolean(args[9]);
    bool              free_edgecolors  = false;

    QuadMeshGenerator path_generator(mesh_width, mesh_height, coordinates.ptr());

    Py::SeqBase<Py::Object> transforms_obj;
    Py::Object edgecolors_obj;
    Py::Tuple linewidths(1);
    linewidths[0] = Py::Float(gc.linewidth);
    Py::SeqBase<Py::Object> linestyles_obj;
    Py::Tuple antialiaseds(1);
    antialiaseds[0] = Py::Int(antialiased ? 1 : 0);

    if (showedges)
    {
        npy_intp dims[] = { 1, 4, 0 };
        double data[] = { 0, 0, 0, 1 };
        edgecolors_obj = Py::Object(PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE,
                                                              (char*)data), true);
    }
    else
    {
        if (antialiased)
        {
            edgecolors_obj = facecolors_obj;
        }
        else
        {
            npy_intp dims[] = { 0, 0 };
            edgecolors_obj = PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
            free_edgecolors = true;
        }
    }

    try
    {
        _draw_path_collection_generic<QuadMeshGenerator, 0, 0>
            (gc,
             master_transform,
             gc.cliprect,
             gc.clippath,
             gc.clippath_trans,
             path_generator,
             transforms_obj,
             offsets_obj,
             offset_trans,
             facecolors_obj,
             edgecolors_obj,
             linewidths,
             linestyles_obj,
             antialiaseds);
    }
    catch (const char* e)
    {
        throw Py::RuntimeError(e);
    }

    return Py::Object();
}

void
RendererAgg::_draw_gouraud_triangle(const double* points,
                                    const double* colors,
                                    agg::trans_affine trans,
                                    bool has_clippath)
{
    typedef agg::rgba8                                         color_t;
    typedef agg::span_gouraud_rgba<color_t>                    span_gen_t;
    typedef agg::span_allocator<color_t>                       span_alloc_t;

    trans *= agg::trans_affine_scaling(1.0, -1.0);
    trans *= agg::trans_affine_translation(0.0, (double)height);

    double tpoints[6];

    for (int i = 0; i < 6; i += 2)
    {
        tpoints[i] = points[i];
        tpoints[i+1] = points[i+1];
        trans.transform(&tpoints[i], &tpoints[i+1]);
    }

    span_alloc_t span_alloc;
    span_gen_t span_gen;

    span_gen.colors(
        agg::rgba(colors[0], colors[1], colors[2], colors[3]),
        agg::rgba(colors[4], colors[5], colors[6], colors[7]),
        agg::rgba(colors[8], colors[9], colors[10], colors[11]));
    span_gen.triangle(
        tpoints[0], tpoints[1],
        tpoints[2], tpoints[3],
        tpoints[4], tpoints[5],
        0.5);

    theRasterizer.add_path(span_gen);

    if (has_clippath)
    {
        typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type> pixfmt_amask_type;
        typedef agg::renderer_base<pixfmt_amask_type>              amask_ren_type;
        typedef agg::renderer_scanline_aa<amask_ren_type, span_alloc_t, span_gen_t>
        amask_aa_renderer_type;

        pixfmt_amask_type pfa(pixFmt, alphaMask);
        amask_ren_type r(pfa);
        amask_aa_renderer_type ren(r, span_alloc, span_gen);
        agg::render_scanlines(theRasterizer, slineP8, ren);
    }
    else
    {
        agg::render_scanlines_aa(theRasterizer, slineP8, rendererBase, span_alloc, span_gen);
    }
}


Py::Object
RendererAgg::draw_gouraud_triangle(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::draw_gouraud_triangle");
    args.verify_length(4);

    GCAgg             gc(args[0], dpi);
    Py::Object        points_obj = args[1];
    Py::Object        colors_obj = args[2];
    agg::trans_affine trans      = py_to_agg_transformation_matrix(args[3].ptr());

    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(gc.cliprect, theRasterizer);
    bool has_clippath = render_clippath(gc.clippath, gc.clippath_trans);

    PyArrayObject* points = (PyArrayObject*)PyArray_ContiguousFromAny
        (points_obj.ptr(), PyArray_DOUBLE, 2, 2);
    if (!points ||
        PyArray_DIM(points, 0) != 3 || PyArray_DIM(points, 1) != 2)
    {
        Py_XDECREF(points);
        throw Py::ValueError("points must be a 3x2 numpy array");
    }
    points_obj = Py::Object((PyObject*)points, true);

    PyArrayObject* colors = (PyArrayObject*)PyArray_ContiguousFromAny
        (colors_obj.ptr(), PyArray_DOUBLE, 2, 2);
    if (!colors ||
        PyArray_DIM(colors, 0) != 3 || PyArray_DIM(colors, 1) != 4)
    {
        Py_XDECREF(colors);
        throw Py::ValueError("colors must be a 3x4 numpy array");
    }
    colors_obj = Py::Object((PyObject*)colors, true);

    _draw_gouraud_triangle(
        (double*)PyArray_DATA(points), (double*)PyArray_DATA(colors),
        trans, has_clippath);

    return Py::Object();
}


Py::Object
RendererAgg::draw_gouraud_triangles(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::draw_gouraud_triangles");
    args.verify_length(4);

    typedef agg::rgba8                      color_t;
    typedef agg::span_gouraud_rgba<color_t> span_gen_t;
    typedef agg::span_allocator<color_t>    span_alloc_t;

    GCAgg             gc(args[0], dpi);
    Py::Object        points_obj = args[1];
    Py::Object        colors_obj = args[2];
    agg::trans_affine trans      = py_to_agg_transformation_matrix(args[3].ptr());
    double            c_points[6];
    double            c_colors[12];

    theRasterizer.reset_clipping();
    rendererBase.reset_clipping(true);
    set_clipbox(gc.cliprect, theRasterizer);
    bool has_clippath = render_clippath(gc.clippath, gc.clippath_trans);

    PyArrayObject* points = (PyArrayObject*)PyArray_FromObject
        (points_obj.ptr(), PyArray_DOUBLE, 3, 3);
    if (!points ||
        PyArray_DIM(points, 1) != 3 || PyArray_DIM(points, 2) != 2)
    {
        Py_XDECREF(points);
        throw Py::ValueError("points must be a Nx3x2 numpy array");
    }
    points_obj = Py::Object((PyObject*)points, true);

    PyArrayObject* colors = (PyArrayObject*)PyArray_FromObject
        (colors_obj.ptr(), PyArray_DOUBLE, 3, 3);
    if (!colors ||
        PyArray_DIM(colors, 1) != 3 || PyArray_DIM(colors, 2) != 4)
    {
        Py_XDECREF(colors);
        throw Py::ValueError("colors must be a Nx3x4 numpy array");
    }
    colors_obj = Py::Object((PyObject*)colors, true);

    if (PyArray_DIM(points, 0) != PyArray_DIM(colors, 0))
    {
        throw Py::ValueError("points and colors arrays must be the same length");
    }

    for (int i = 0; i < PyArray_DIM(points, 0); ++i)
    {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 2; ++k) {
                c_points[j*2+k] = *(double *)PyArray_GETPTR3(points, i, j, k);
            }
        }

        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                c_colors[j*4+k] = *(double *)PyArray_GETPTR3(colors, i, j, k);
            }
        }

        _draw_gouraud_triangle(
                c_points, c_colors, trans, has_clippath);
    }

    return Py::Object();
}


Py::Object
RendererAgg::write_rgba(const Py::Tuple& args)
{
    _VERBOSE("RendererAgg::write_rgba");

    args.verify_length(1);

    FILE *fp = NULL;
    bool close_file = false;
    Py::Object py_fileobj = Py::Object(args[0]);

    #if PY3K
    int fd = PyObject_AsFileDescriptor(py_fileobj.ptr());
    PyErr_Clear();
    #endif

    if (py_fileobj.isString())
    {
        std::string fileName = Py::String(py_fileobj);
        const char *file_name = fileName.c_str();
        if ((fp = fopen(file_name, "wb")) == NULL)
            throw Py::RuntimeError(
                Printf("Could not open file %s", file_name).str());
        if (fwrite(pixBuffer, 1, NUMBYTES, fp) != NUMBYTES)
        {
            fclose(fp);
            throw Py::RuntimeError(
                Printf("Error writing to file %s", file_name).str());
        }
        close_file = true;
    }
    #if PY3K
    else if (fd != -1)
    {
        if (write(fd, pixBuffer, NUMBYTES) != (ssize_t)NUMBYTES)
        {
            throw Py::RuntimeError("Error writing to file");
        }
    }
    #else
    else if (PyFile_CheckExact(py_fileobj.ptr()))
    {
        fp = PyFile_AsFile(py_fileobj.ptr());
        if (fwrite(pixBuffer, 1, NUMBYTES, fp) != NUMBYTES)
        {
            throw Py::RuntimeError("Error writing to file");
        }
    }
    #endif
    else
    {
        PyObject* write_method = PyObject_GetAttrString(py_fileobj.ptr(),
                                                        "write");
        if (!(write_method && PyCallable_Check(write_method)))
        {
            Py_XDECREF(write_method);
            throw Py::TypeError(
                "Object does not appear to be a 8-bit string path or a Python file-like object");
        }

        PyObject_CallFunction(write_method, (char *)"s#", pixBuffer, NUMBYTES);

        Py_XDECREF(write_method);
    }

    return Py::Object();
}


Py::Object
RendererAgg::tostring_rgb(const Py::Tuple& args)
{
    //"Return the rendered buffer as an RGB string";

    _VERBOSE("RendererAgg::tostring_rgb");

    args.verify_length(0);
    int row_len = width * 3;
    unsigned char* buf_tmp = new unsigned char[row_len * height];
    if (buf_tmp == NULL)
    {
        //todo: also handle allocation throw
        throw Py::MemoryError(
            "RendererAgg::tostring_rgb could not allocate memory");
    }

    try
    {
        agg::rendering_buffer renderingBufferTmp;
        renderingBufferTmp.attach(buf_tmp,
                                  width,
                                  height,
                                  row_len);

        agg::color_conv(&renderingBufferTmp, &renderingBuffer,
                        agg::color_conv_rgba32_to_rgb24());
    }
    catch (...)
    {
        delete [] buf_tmp;
        throw Py::RuntimeError("Unknown exception occurred in tostring_rgb");
    }

    //todo: how to do this with native CXX
    #if PY3K
    PyObject* o = Py_BuildValue("y#", buf_tmp, row_len * height);
    #else
    PyObject* o = Py_BuildValue("s#", buf_tmp, row_len * height);
    #endif

    delete [] buf_tmp;
    return Py::asObject(o);
}


Py::Object
RendererAgg::tostring_argb(const Py::Tuple& args)
{
    //"Return the rendered buffer as an RGB string";

    _VERBOSE("RendererAgg::tostring_argb");

    args.verify_length(0);
    int row_len = width * 4;
    unsigned char* buf_tmp = new unsigned char[row_len * height];
    if (buf_tmp == NULL)
    {
        //todo: also handle allocation throw
        throw Py::MemoryError("RendererAgg::tostring_argb could not allocate memory");
    }

    try
    {
        agg::rendering_buffer renderingBufferTmp;
        renderingBufferTmp.attach(buf_tmp, width, height, row_len);
        agg::color_conv(&renderingBufferTmp, &renderingBuffer, agg::color_conv_rgba32_to_argb32());
    }
    catch (...)
    {
        delete [] buf_tmp;
        throw Py::RuntimeError("Unknown exception occurred in tostring_argb");
    }

    //todo: how to do this with native CXX

    #if PY3K
    PyObject* o = Py_BuildValue("y#", buf_tmp, row_len * height);
    #else
    PyObject* o = Py_BuildValue("s#", buf_tmp, row_len * height);
    #endif
    delete [] buf_tmp;
    return Py::asObject(o);
}


Py::Object
RendererAgg::tostring_bgra(const Py::Tuple& args)
{
    //"Return the rendered buffer as an RGB string";

    _VERBOSE("RendererAgg::tostring_bgra");

    args.verify_length(0);
    int row_len = width * 4;
    unsigned char* buf_tmp = new unsigned char[row_len * height];
    if (buf_tmp == NULL)
    {
        //todo: also handle allocation throw
        throw Py::MemoryError("RendererAgg::tostring_bgra could not allocate memory");
    }

    try
    {
        agg::rendering_buffer renderingBufferTmp;
        renderingBufferTmp.attach(buf_tmp,
                                  width,
                                  height,
                                  row_len);

        agg::color_conv(&renderingBufferTmp, &renderingBuffer, agg::color_conv_rgba32_to_bgra32());
    }
    catch (...)
    {
        delete [] buf_tmp;
        throw Py::RuntimeError("Unknown exception occurred in tostring_bgra");
    }

    //todo: how to do this with native CXX
    #if PY3K
    PyObject* o = Py_BuildValue("y#", buf_tmp, row_len * height);
    #else
    PyObject* o = Py_BuildValue("s#", buf_tmp, row_len * height);
    #endif
    delete [] buf_tmp;
    return Py::asObject(o);
}


Py::Object
RendererAgg::buffer_rgba(const Py::Tuple& args)
{
    //"expose the rendered buffer as Python buffer object, starting from postion x,y";

    _VERBOSE("RendererAgg::buffer_rgba");

    args.verify_length(0);

    #if PY3K
    return Py::asObject(this);
    #else
    int row_len = width * 4;
    return Py::asObject(PyBuffer_FromMemory(pixBuffer, row_len*height));
    #endif
}


Py::Object
RendererAgg::tostring_rgba_minimized(const Py::Tuple& args)
{
    args.verify_length(0);

    int xmin = width;
    int ymin = height;
    int xmax = 0;
    int ymax = 0;

    // Looks at the alpha channel to find the minimum extents of the image
    unsigned char* pixel = pixBuffer + 3;
    for (int y = 0; y < (int)height; ++y)
    {
        for (int x = 0; x < (int)width; ++x)
        {
            if (*pixel)
            {
                if (x < xmin) xmin = x;
                if (y < ymin) ymin = y;
                if (x > xmax) xmax = x;
                if (y > ymax) ymax = y;
            }
            pixel += 4;
        }
    }

    int newwidth = 0;
    int newheight = 0;
    Py::String data;
    if (xmin < xmax && ymin < ymax)
    {
        // Expand the bounds by 1 pixel on all sides
        xmin = std::max(0, xmin - 1);
        ymin = std::max(0, ymin - 1);
        xmax = std::min(xmax, (int)width);
        ymax = std::min(ymax, (int)height);

        newwidth    = xmax - xmin;
        newheight   = ymax - ymin;
        int newsize = newwidth * newheight * 4;

        unsigned char* buf = new unsigned char[newsize];
        if (buf == NULL)
        {
            throw Py::MemoryError("RendererAgg::tostring_minimized could not allocate memory");
        }

        unsigned int*  dst = (unsigned int*)buf;
        unsigned int*  src = (unsigned int*)pixBuffer;
        for (int y = ymin; y < ymax; ++y)
        {
            for (int x = xmin; x < xmax; ++x, ++dst)
            {
                *dst = src[y * width + x];
            }
        }

        // The Py::String will take over the buffer
        data = Py::String((const char *)buf, (int)newsize);
    }

    Py::Tuple bounds(4);
    bounds[0] = Py::Int(xmin);
    bounds[1] = Py::Int(ymin);
    bounds[2] = Py::Int(newwidth);
    bounds[3] = Py::Int(newheight);

    Py::Tuple result(2);
    result[0] = data;
    result[1] = bounds;

    return result;
}


Py::Object
RendererAgg::clear(const Py::Tuple& args)
{
    //"clear the rendered buffer";

    _VERBOSE("RendererAgg::clear");

    args.verify_length(0);
    rendererBase.clear(agg::rgba(1, 1, 1, 0));

    return Py::Object();
}


agg::rgba
RendererAgg::rgb_to_color(const Py::SeqBase<Py::Object>& rgb, double alpha)
{
    _VERBOSE("RendererAgg::rgb_to_color");

    double r = Py::Float(rgb[0]);
    double g = Py::Float(rgb[1]);
    double b = Py::Float(rgb[2]);
    return agg::rgba(r, g, b, alpha);
}


double
RendererAgg::points_to_pixels(const Py::Object& points)
{
    _VERBOSE("RendererAgg::points_to_pixels");
    double p = Py::Float(points) ;
    return p * dpi / 72.0;
}

#if PY3K
int
RendererAgg::buffer_get( Py_buffer* buf, int flags )
{
    return PyBuffer_FillInfo(buf, this, pixBuffer, width * height * 4, 1,
                             PyBUF_SIMPLE);
}
#endif

/* ------------ module methods ------------- */
Py::Object _backend_agg_module::new_renderer(const Py::Tuple &args,
        const Py::Dict &kws)
{

    if (args.length() != 3)
    {
        throw Py::RuntimeError("Incorrect # of args to RendererAgg(width, height, dpi).");
    }

    int debug;
    if (kws.hasKey("debug"))
    {
        debug = Py::Int(kws["debug"]);
    }
    else
    {
        debug = 0;
    }

    unsigned int width = (int)Py::Int(args[0]);
    unsigned int height = (int)Py::Int(args[1]);
    double dpi = Py::Float(args[2]);

    if (width > 1 << 15 || height > 1 << 15)
    {
        throw Py::ValueError("width and height must each be below 32768");
    }

    if (dpi <= 0.0)
    {
        throw Py::ValueError("dpi must be positive");
    }

    RendererAgg* renderer = NULL;
    try
    {
        renderer = new RendererAgg(width, height, dpi, debug);
    }
    catch (std::bad_alloc)
    {
        throw Py::RuntimeError("Could not allocate memory for image");
    }

    return Py::asObject(renderer);
}


void BufferRegion::init_type()
{
    behaviors().name("BufferRegion");
    behaviors().doc("A wrapper to pass agg buffer objects to and from the python level");


    add_varargs_method("set_x", &BufferRegion::set_x,
                       "set_x(x)");

    add_varargs_method("set_y", &BufferRegion::set_y,
                       "set_y(y)");

    add_varargs_method("get_extents", &BufferRegion::get_extents,
                       "get_extents()");

    add_varargs_method("to_string", &BufferRegion::to_string,
                       "to_string()");
    add_varargs_method("to_string_argb", &BufferRegion::to_string_argb,
                       "to_string_argb()");
}


void RendererAgg::init_type()
{
    behaviors().name("RendererAgg");
    behaviors().doc("The agg backend extension module");

    add_varargs_method("draw_path", &RendererAgg::draw_path,
                       "draw_path(gc, path, transform, rgbFace)\n");
    add_varargs_method("draw_path_collection", &RendererAgg::draw_path_collection,
                       "draw_path_collection(gc, master_transform, paths, transforms, offsets, offsetTrans, facecolors, edgecolors, linewidths, linestyles, antialiaseds)\n");
    add_varargs_method("draw_quad_mesh", &RendererAgg::draw_quad_mesh,
                       "draw_quad_mesh(gc, master_transform, meshWidth, meshHeight, coordinates, offsets, offsetTrans, facecolors, antialiaseds, showedges)\n");
    add_varargs_method("draw_gouraud_triangle", &RendererAgg::draw_gouraud_triangle,
                       "draw_gouraud_triangle(gc, points, colors, master_transform)\n");
    add_varargs_method("draw_gouraud_triangles", &RendererAgg::draw_gouraud_triangles,
                       "draw_gouraud_triangles(gc, points, colors, master_transform)\n");
    add_varargs_method("draw_markers", &RendererAgg::draw_markers,
                       "draw_markers(gc, marker_path, marker_trans, path, rgbFace)\n");
    add_varargs_method("draw_text_image", &RendererAgg::draw_text_image,
                       "draw_text_image(font_image, x, y, r, g, b, a)\n");
    add_varargs_method("draw_image", &RendererAgg::draw_image,
                       "draw_image(gc, x, y, im)");
    add_varargs_method("write_rgba", &RendererAgg::write_rgba,
                       "write_rgba(fname)");
    add_varargs_method("tostring_rgb", &RendererAgg::tostring_rgb,
                       "s = tostring_rgb()");
    add_varargs_method("tostring_argb", &RendererAgg::tostring_argb,
                       "s = tostring_argb()");
    add_varargs_method("tostring_bgra", &RendererAgg::tostring_bgra,
                       "s = tostring_bgra()");
    add_varargs_method("tostring_rgba_minimized", &RendererAgg::tostring_rgba_minimized,
                       "s = tostring_rgba_minimized()");
    add_varargs_method("buffer_rgba", &RendererAgg::buffer_rgba,
                       "buffer = buffer_rgba()");
    add_varargs_method("clear", &RendererAgg::clear,
                       "clear()");
    add_varargs_method("copy_from_bbox", &RendererAgg::copy_from_bbox,
                       "copy_from_bbox(bbox)");
    add_varargs_method("restore_region", &RendererAgg::restore_region,
                       "restore_region(region)");
    add_varargs_method("restore_region2", &RendererAgg::restore_region2,
                       "restore_region(region, x1, y1, x2, y2, x3, y3)");

    #if PY3K
    behaviors().supportBufferType();
    #endif
}
