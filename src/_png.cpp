/* -*- mode: c++; c-basic-offset: 4 -*- */

/* For linux, png.h must be imported before Python.h because
   png.h needs to be the one to define setjmp.
   Undefining _POSIX_C_SOURCE and _XOPEN_SOURCE stops a couple
   of harmless warnings.
*/

#include "_png.h"
#include <png.h>

// As reported in [3082058] build _png.so on aix
#ifdef _AIX
#undef jmpbuf
#endif

struct method_obj {
    void* obj;
    data_write_fn write_fn;
    data_flush_fn flush_fn;
    data_read_fn read_fn;
};

static void
write_png_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    method_obj* method = (method_obj*)png_get_io_ptr(png_ptr);
    method->write_fn(method->obj, (unsigned char *)data, (size_t)length);
}

static void
flush_png_data(png_structp png_ptr)
{
    method_obj* method = (method_obj*)png_get_io_ptr(png_ptr);
    method->flush_fn(method->obj);
}

static void
read_png_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    method_obj* method = (method_obj*)png_get_io_ptr(png_ptr);
    method->read_fn(method->obj, (unsigned char *)data, (size_t)length);
}

void
write_png(const void* const buffer, const int width, const int height,
          FILE* fp,
          void* const state, data_write_fn write_fn,
          data_flush_fn flush_fn,
          const double dpi)
{
    png_bytep pixbuf = NULL;
    png_bytep *row_pointers = NULL;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;

    try {
        png_uint_32 row;
        pixbuf = (png_bytep)buffer;
        row_pointers = new png_bytep[height];
        for (row = 0; row < (png_uint_32)height; ++row)
        {
            row_pointers[row] = pixbuf + row * width * 4;
        }

        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (png_ptr == NULL)
        {
            throw "Could not create write struct";
        }

        info_ptr = png_create_info_struct(png_ptr);
        if (info_ptr == NULL)
        {
            throw "Could not create info struct";
        }

        if (setjmp(png_jmpbuf(png_ptr)))
        {
            throw "Error building image";
        }

        if (fp)
        {
            png_init_io(png_ptr, fp);
        }
        else
        {
            /* Write to a Python function */
            method_obj method;
            method.obj = state;
            method.write_fn = write_fn;
            method.flush_fn = flush_fn;
            png_set_write_fn(
                png_ptr, (void *)&method, &write_png_data, &flush_png_data);
        }
        png_set_IHDR(png_ptr, info_ptr,
                     width, height, 8,
                     PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        // Save the dpi of the image in the file
        if (dpi > 0.0)
        {
            size_t dots_per_meter = (size_t)(dpi / (2.54 / 100.0));
            png_set_pHYs(png_ptr, info_ptr, dots_per_meter, dots_per_meter,
                         PNG_RESOLUTION_METER);
        }

        // this a a color image!
        struct png_color_8_struct sig_bit;
        sig_bit.gray = 0;
        sig_bit.red = 8;
        sig_bit.green = 8;
        sig_bit.blue = 8;
        /* if the image has an alpha channel then */
        sig_bit.alpha = 8;
        png_set_sBIT(png_ptr, info_ptr, &sig_bit);

        png_write_info(png_ptr, info_ptr);
        png_write_image(png_ptr, row_pointers);
        png_write_end(png_ptr, info_ptr);
    }
    catch (...)
    {
        if (png_ptr && info_ptr)
        {
            png_destroy_write_struct(&png_ptr, &info_ptr);
        }
        delete [] row_pointers;
        throw;
    }

    if (png_ptr && info_ptr)
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
    }
    delete [] row_pointers;
}

void
read_png_header(FILE* fp, void* const state,
                data_read_fn read_fn, readinfo* const info)
{
    png_byte header[8];   // 8 is the maximum size that can be checked
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;

    info->png_ptr = NULL;
    info->info_ptr = NULL;

    if (fp)
    {
        if (fread(header, 1, 8, fp) != 8)
        {
            throw "Error reading PNG header";
        }
    }
    else
    {
        read_fn(state, header, 8);
    }

    if (png_sig_cmp(header, 0, 8))
    {
        throw "File not recognized as a PNG file";
    }

    /* initialize stuff */
    try {
        png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!png_ptr)
        {
            throw "png_create_read_struct failed";
        }

        info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr)
        {
            throw "png_create_info_struct failed";
        }

        if (setjmp(png_jmpbuf(png_ptr)))
        {
            throw "error building image";
        }

        if (fp)
        {
            png_init_io(png_ptr, fp);
        }
        else
        {
            /* Read from a Python function */
            method_obj method;
            method.obj = state;
            method.read_fn = read_fn;
            png_set_read_fn(png_ptr, (void*)&method, &read_png_data);
        }
        png_set_sig_bytes(png_ptr, 8);
        png_read_info(png_ptr, info_ptr);

        info->width = png_get_image_width(png_ptr, info_ptr);
        info->height = png_get_image_height(png_ptr, info_ptr);
        if (png_get_color_type(png_ptr, info_ptr) & PNG_COLOR_MASK_ALPHA)
        {
            info->planes = 4;     //RGBA images
        }
        else if (png_get_color_type(png_ptr, info_ptr) & PNG_COLOR_MASK_COLOR)
        {
            info->planes = 3;     //RGB images
        }
        else
        {
            info->planes = 1;     //Greyscale images
        }
    } catch (...) {
        if (png_ptr && info_ptr) {
#ifndef png_infopp_NULL
            png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
#else
            png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
#endif
        }
        throw;
    }

    info->png_ptr = png_ptr;
    info->info_ptr = info_ptr;
}

void
read_png_content(FILE* fp, void* const state,
                 data_read_fn read_fn,
                 void* buffer, const int datatype,
                 const readinfo* const info)
{
    png_structp png_ptr = (png_structp)info->png_ptr;
    png_infop info_ptr = (png_infop)info->info_ptr;
    png_bytep* row_pointers;
    int row;
    int width = info->width;
    int height = info->height;
    int planes = info->planes;

    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // Unpack 1, 2, and 4-bit images
    if (bit_depth < 8)
    {
        png_set_packing(png_ptr);
    }

    // If sig bits are set, shift data
    png_color_8p sig_bit;
    if ((png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_PALETTE) &&
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
    if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_PALETTE)
    {
        png_set_palette_to_rgb(png_ptr);
    }

    // If there's an alpha channel convert gray to RGB
    if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_GRAY_ALPHA)
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

    if (datatype == -32) {
        double max_value = (1 << ((bit_depth < 8) ? 8 : bit_depth)) - 1;
        float* out = (float*)buffer;

        for (int y = 0; y < height; y++)
        {
            png_byte* row = row_pointers[y];
            for (int x = 0; x < width; x++)
            {
                size_t offset = (y * width + x) * planes;
                if (bit_depth == 16)
                {
                    png_uint_16* ptr = &reinterpret_cast<png_uint_16*>(row)[x * planes];
                    for (int p = 0; p < planes; p++)
                    {
                        *(out + offset + p) = (float)(ptr[p]) / max_value;
                    }
                }
                else
                {
                    png_byte* ptr = &(row[x * planes]);
                    for (int p = 0; p < planes; p++)
                    {
                        *(out + offset + p) = (float)(ptr[p]) / max_value;
                    }
                }
            }
        }
    } else if (datatype == 8) {
        png_bytep out = (png_bytep)buffer;

        for (int y = 0; y < height; y++)
        {
            png_byte* row = row_pointers[y];
            for (int x = 0; x < width; x++)
            {
                size_t offset = (y * width + x) * planes;
                if (bit_depth == 16)
                {
                    png_uint_16* ptr = &reinterpret_cast<png_uint_16*>(row)[x * planes];
                    for (int p = 0; p < planes; p++)
                    {
                        *(out + offset + p) = ptr[p] >> 8;
                    }
                }
                else
                {
                    png_byte* ptr = &(row[x * planes]);
                    for (int p = 0; p < planes; p++)
                    {
                        *(out + offset + p) = ptr[p];
                    }
                }
            }
        }
    }

    //free the png memory
    png_read_end(png_ptr, info_ptr);
#ifndef png_infopp_NULL
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
#else
    png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
#endif
    for (row = 0; row < height; row++)
    {
        delete [] row_pointers[row];
    }
    delete [] row_pointers;
}
