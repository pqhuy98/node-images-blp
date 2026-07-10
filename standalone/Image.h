/*
 * Standalone Image.h for native BLP encoder — no Node/N-API dependency.
 */
#ifndef __NODE_IMAGE_H__
#define __NODE_IMAGE_H__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

typedef enum {
    TYPE_PNG = 1,
    TYPE_JPEG,
    TYPE_GIF,
    TYPE_BMP,
    TYPE_BLP,
    TYPE_RAW,
    TYPE_WEBP,
} ImageType;

typedef enum {
    FAIL = 0,
    SUCCESS,
} ImageState;

typedef struct Pixel {
    uint8_t R;
    uint8_t G;
    uint8_t B;
    uint8_t A;

    void Merge(struct Pixel *pixel);
} Pixel;

typedef enum {
    EMPTY = 0,
    ALPHA,
    SOLID,
} PixelArrayType;

typedef struct PixelArray {
    Pixel **data;
    size_t width;
    size_t height;
    PixelArrayType type;

    int32_t Size() {
        return (int32_t)((height * sizeof(Pixel **)) + (width * height * sizeof(Pixel)));
    }

    ImageState Malloc(size_t w, size_t h);
    ImageState CopyFrom(struct PixelArray *src, size_t x, size_t y, size_t w, size_t h);
    void Free();
    void Draw(struct PixelArray *src, size_t x, size_t y);
    void Fill(Pixel *color);
    ImageState SetWidth(size_t w);
    ImageState SetHeight(size_t h);
    ImageState Resize(size_t w, size_t h, const char *filter);
    ImageState Rotate(size_t deg);
    void DetectTransparent();
} PixelArray;

typedef struct {
    uint8_t *data;
    unsigned long length;
    unsigned long position;
} ImageData;

typedef struct {
    char *data;
    unsigned long length;
} ImageConfig;

typedef ImageState (*ImageEncoder)(PixelArray *input, ImageData *output, ImageConfig *config);
typedef ImageState (*ImageDecoder)(PixelArray *output, ImageData *input);

typedef struct ImageCodec {
    ImageType type;
    ImageEncoder encoder;
    ImageDecoder decoder;
    struct ImageCodec *next;
} ImageCodec;

#define ENCODER(type) encode ## type
#define ENCODER_FN(type) ImageState ENCODER(type)(PixelArray *input, ImageData *output, ImageConfig *config)
#define DECODER(type) decode ## type
#define DECODER_FN(type) ImageState DECODER(type)(PixelArray *output, ImageData *input)
#define IMAGE_CODEC(type) DECODER_FN(type); ENCODER_FN(type)

#ifdef HAVE_PNG
IMAGE_CODEC(Png);
#endif

#ifdef HAVE_BLP
IMAGE_CODEC(Blp);
#endif

#endif
