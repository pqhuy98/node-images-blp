#include "Image.h"

#include <string.h>

static const size_t kMaxImageWidth = 16384;
static const size_t kMaxImageHeight = 16384;

void Pixel::Merge(Pixel *pixel) {
    double a, af, ab;
    ab = (double)A / 0xFF;
    af = (double)pixel->A / 0xFF;
    a = (1 - (1 - af) * (1 - ab));

    R = (uint8_t)((pixel->R * af + R * ab * (1 - af)) / a);
    G = (uint8_t)((pixel->G * af + G * ab * (1 - af)) / a);
    B = (uint8_t)((pixel->B * af + B * ab * (1 - af)) / a);
    A = (uint8_t)(a * 0xFF);
}

ImageState PixelArray::Malloc(size_t w, size_t h) {
    int32_t size;
    Pixel *line;

    if (w > 0 && h > 0) {
        if (w > kMaxImageWidth || h > kMaxImageHeight) {
            goto fail;
        }

        if ((data = (Pixel **)malloc(h * sizeof(Pixel **))) == NULL) {
            goto fail;
        }

        width = w;
        size = (int32_t)(width * sizeof(Pixel));
        for (height = 0; height < h; height++) {
            if ((line = (Pixel *)malloc((size_t)size)) == NULL) {
                goto free;
            }
            memset(line, 0x00, (size_t)size);
            data[height] = line;
        }
    }
    return SUCCESS;

free:
    while (height--) {
        free(data[height]);
    }
    free(data);

fail:
    width = height = 0;
    type = EMPTY;
    data = NULL;
    return FAIL;
}

void PixelArray::Free() {
    if (data != NULL) {
        size_t h = height;
        while (h--) {
            if (data[h] != NULL) {
                free(data[h]);
            }
        }
        free(data);
    }

    width = height = 0;
    type = EMPTY;
    data = NULL;
}

void PixelArray::DetectTransparent() {
    size_t x, y;
    Pixel *pixel;
    bool empty, opaque, alpha;
    type = EMPTY;

    empty = opaque = alpha = false;

    for (y = 0; y < height; y++) {
        pixel = data[y];
        for (x = 0; x < width; x++, pixel++) {
            switch (pixel->A) {
                case 0x00:
                    empty = true;
                    break;
                case 0xFF:
                    opaque = true;
                    break;
                default:
                    alpha = true;
                    break;
            }

            if (alpha || (empty && opaque)) {
                type = ALPHA;
                return;
            }
        }
    }
    type = opaque ? SOLID : EMPTY;
}

ImageState PixelArray::CopyFrom(PixelArray *, size_t, size_t, size_t, size_t) { return FAIL; }
void PixelArray::Draw(PixelArray *, size_t, size_t) {}
void PixelArray::Fill(Pixel *) {}
ImageState PixelArray::SetWidth(size_t) { return FAIL; }
ImageState PixelArray::SetHeight(size_t) { return FAIL; }
ImageState PixelArray::Resize(size_t, size_t, const char *) { return FAIL; }
ImageState PixelArray::Rotate(size_t) { return FAIL; }
