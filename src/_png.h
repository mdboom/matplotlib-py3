#ifndef ___PNG_H__
#define ___PNG_H__

#include <stdio.h>

typedef void(*data_write_fn)(void*, unsigned char*, size_t);
typedef void(*data_flush_fn)(void*);
typedef void(*data_read_fn)(void*, unsigned char*, size_t);

void
write_png(const void* const buffer, const int width, const int height,
          FILE* fp,
          void* const state, data_write_fn write_fn,
          data_flush_fn flush_fn,
          const double dpi);

typedef struct readinfo {
    int width;
    int height;
    int planes;
    void* png_ptr;
    void* info_ptr;
};

void
read_png_header(FILE* fp, void* const state,
                data_read_fn read_fn, readinfo* const info);

void
read_png_content(FILE* fp, void* const state,
                 data_read_fn read_fn,
                 void* buffer, const int datatype,
                 const readinfo* const info);

#endif
