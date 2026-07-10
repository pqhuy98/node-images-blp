#ifndef BLP_ENCODE_API_H
#define BLP_ENCODE_API_H

#include <stddef.h>

#if defined(_WIN32)
  #if defined(BLPENCODE_EXPORTS)
    #define BLPENCODE_API __declspec(dllexport)
  #else
    #define BLPENCODE_API __declspec(dllimport)
  #endif
#else
  #define BLPENCODE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

BLPENCODE_API int blp_encode_png(const unsigned char *png, size_t png_len, unsigned char **out_buf, size_t *out_len);
BLPENCODE_API void blp_encode_free(unsigned char *buf);

#ifdef __cplusplus
}
#endif

#endif
