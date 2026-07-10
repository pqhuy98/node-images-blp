#define BLPENCODE_EXPORTS
#include "blp_encode_api.h"
#include "Image.h"

#include <string.h>

extern "C" BLPENCODE_API int blp_encode_png(const unsigned char *png, size_t png_len, unsigned char **out_buf, size_t *out_len) {
    if (!png || png_len == 0 || !out_buf || !out_len) {
        return 1;
    }

    *out_buf = NULL;
    *out_len = 0;

    PixelArray pixels;
    memset(&pixels, 0, sizeof(pixels));

    ImageData input;
    memset(&input, 0, sizeof(input));
    input.data = (uint8_t *)png;
    input.length = (unsigned long)png_len;
    input.position = 0;

    if (decodePng(&pixels, &input) != SUCCESS) {
        return 2;
    }

    ImageData output;
    memset(&output, 0, sizeof(output));
    ImageConfig config;
    memset(&config, 0, sizeof(config));

    ImageState state = encodeBlp(&pixels, &output, &config);
    pixels.Free();
    if (state != SUCCESS || output.data == NULL || output.length == 0) {
        if (output.data != NULL) {
            free(output.data);
        }
        return 3;
    }

    *out_buf = output.data;
    *out_len = (size_t)output.length;
    return 0;
}

extern "C" BLPENCODE_API void blp_encode_free(unsigned char *buf) {
    if (buf != NULL) {
        free(buf);
    }
}
