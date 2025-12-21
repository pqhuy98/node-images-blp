/*
 * Blp.cc - Minimal BLP1 (palette + 8-bit alpha) encoder/decoder stub
 */

#include "Image.h"

#ifdef HAVE_BLP

#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>

// --- Helpers ---------------------------------------------------------------
static void writeUInt32LE(uint8_t *dst, size_t offset, uint32_t value) {
    dst[offset + 0] = (uint8_t)(value & 0xFF);
    dst[offset + 1] = (uint8_t)((value >> 8) & 0xFF);
    dst[offset + 2] = (uint8_t)((value >> 16) & 0xFF);
    dst[offset + 3] = (uint8_t)((value >> 24) & 0xFF);
}

struct RgbF { float r, g, b; };

static inline float dist2_srgb(float r1, float g1, float b1, float r2, float g2, float b2) {
    // Perceptual weights (BT.709) in gamma-coded space (sRGB-like)
    const float WR = 0.2126f, WG = 0.7152f, WB = 0.0722f;
    float dr = r1 - r2;
    float dg = g1 - g2;
    float db = b1 - b2;
    return WR * dr * dr + WG * dg * dg + WB * db * db;
}

static inline int lutIndex64(int r6, int g6, int b6) {
    return (r6 << 12) | (g6 << 6) | b6;
}

static void buildLut64FromPalette(const uint8_t *palette /* 256*4 BGRA */, size_t paletteSize,
                                 std::vector<RgbF> &paletteRgb,
                                 std::vector<uint8_t> &lut /* 64^3 */) {
    paletteRgb.clear();
    paletteRgb.reserve(std::max((size_t)1, paletteSize));

    size_t count = paletteSize;
    if (count == 0) count = 1;
    for (size_t i = 0; i < count; i++) {
        size_t p = i * 4;
        const float b = (float)palette[p + 0];
        const float g = (float)palette[p + 1];
        const float r = (float)palette[p + 2];
        paletteRgb.push_back(RgbF{ r, g, b });
    }

    static const int LUTN = 64; // 6 bits/channel
    const int LUT_SIZE = LUTN * LUTN * LUTN;
    lut.assign(LUT_SIZE, 0);

    for (int rr = 0; rr < LUTN; rr++) {
        float r_srgb = (float)((rr << 2) | 2); // bucket midpoint in 0..255
        for (int gg = 0; gg < LUTN; gg++) {
            float g_srgb = (float)((gg << 2) | 2);
            for (int bb = 0; bb < LUTN; bb++) {
                float b_srgb = (float)((bb << 2) | 2);
                int best = 0;
                float bestD = 1e30f;
                for (size_t k = 0; k < paletteRgb.size(); k++) {
                    const auto &c = paletteRgb[k];
                    float d = dist2_srgb(r_srgb, g_srgb, b_srgb, c.r, c.g, c.b);
                    if (d < bestD) { bestD = d; best = (int)k; }
                }
                lut[lutIndex64(rr, gg, bb)] = (uint8_t)best;
            }
        }
    }
}

static void mapToPaletteNoDither(PixelArray *input,
                                const std::vector<uint8_t> &lut /* 64^3 */,
                                uint8_t *indices /* width*height */,
                                uint8_t *alpha /* width*height */) {
    const size_t w = input->width;
    const size_t h = input->height;
    for (size_t y = 0; y < h; y++) {
        Pixel *row = input->data[y];
        for (size_t x = 0; x < w; x++) {
            Pixel p = row[x];
            alpha[y * w + x] = p.A;
            const int r6 = ((int)p.R) >> 2;
            const int g6 = ((int)p.G) >> 2;
            const int b6 = ((int)p.B) >> 2;
            indices[y * w + x] = lut[lutIndex64(r6, g6, b6)];
        }
    }
}

// Downsample by 2 with a simple 2x2 box filter, treating alpha as a separate mask channel:
// - RGB is averaged without alpha-weighting
// - Alpha is averaged independently
// This matches the "resize RGB vs alpha separately" behavior used elsewhere in the toolchain.
static ImageState downsample2x2SeparateAlpha(const PixelArray *src, PixelArray *dst) {
    if (!src || !dst || !src->data || !dst->data) return FAIL;
    if (src->width == 0 || src->height == 0 || dst->width == 0 || dst->height == 0) return FAIL;

    const size_t sw = src->width;
    const size_t sh = src->height;
    const size_t dw = dst->width;
    const size_t dh = dst->height;

    for (size_t y = 0; y < dh; y++) {
        for (size_t x = 0; x < dw; x++) {
            const size_t sx0 = std::min(sw - 1, x * 2);
            const size_t sy0 = std::min(sh - 1, y * 2);
            const size_t sx1 = std::min(sw - 1, sx0 + 1);
            const size_t sy1 = std::min(sh - 1, sy0 + 1);

            const Pixel p00 = src->data[sy0][sx0];
            const Pixel p10 = src->data[sy0][sx1];
            const Pixel p01 = src->data[sy1][sx0];
            const Pixel p11 = src->data[sy1][sx1];

            const uint32_t sumA = (uint32_t)p00.A + (uint32_t)p10.A + (uint32_t)p01.A + (uint32_t)p11.A;
            const uint32_t sumR = (uint32_t)p00.R + (uint32_t)p10.R + (uint32_t)p01.R + (uint32_t)p11.R;
            const uint32_t sumG = (uint32_t)p00.G + (uint32_t)p10.G + (uint32_t)p01.G + (uint32_t)p11.G;
            const uint32_t sumB = (uint32_t)p00.B + (uint32_t)p10.B + (uint32_t)p01.B + (uint32_t)p11.B;

            uint32_t outA = (sumA + 2) / 4; // average alpha (rounded)
            uint32_t outR = (sumR + 2) / 4;
            uint32_t outG = (sumG + 2) / 4;
            uint32_t outB = (sumB + 2) / 4;

            Pixel &d = dst->data[y][x];
            d.R = (uint8_t)std::min(255u, outR);
            d.G = (uint8_t)std::min(255u, outG);
            d.B = (uint8_t)std::min(255u, outB);
            d.A = (uint8_t)std::min(255u, outA);
        }
    }
    return SUCCESS;
}

// High-quality, fast quantization: k-means on sampled pixels + 5-5-5 LUT mapping
// - Operates in linear RGB for perceptual accuracy (convert from sRGB)
// - Sample up to SAMPLE_MAX non-transparent pixels
// - Run limited-iteration k-means (K<=256)
// - Build 32x32x32 (5 bits/channel) LUT for fast full-frame mapping
static inline float srgbToLinear(float c01) {
    // c01 in [0,1]
    if (c01 <= 0.04045f) return c01 / 12.92f;
    return powf((c01 + 0.055f) / 1.055f, 2.4f);
}

static inline uint8_t linearToSrgb8(float cLin) {
    // clamp
    if (cLin < 0.0f) cLin = 0.0f; else if (cLin > 1.0f) cLin = 1.0f;
    float c;
    if (cLin <= 0.0031308f) c = 12.92f * cLin; else c = 1.055f * powf(cLin, 1.0f / 2.4f) - 0.055f;
    int v = (int)floorf(c * 255.0f + 0.5f);
    if (v < 0) v = 0; else if (v > 255) v = 255;
    return (uint8_t)v;
}
static void quantize_kmeans(PixelArray *input,
                            uint8_t *palette /* 256*4 BGRA */, size_t *paletteSize,
                            uint8_t *indices /* width*height */,
                            uint8_t *alpha /* width*height */) {
    const size_t w = input->width;
    const size_t h = input->height;
    const size_t n = w * h;

    // Perceptual weights (BT.709) in gamma-coded space (sRGB-like)
    const float WR = 0.2126f, WG = 0.7152f, WB = 0.0722f;
    auto dist2 = [&](float r1, float g1, float b1, float r2, float g2, float b2) {
        float dr = r1 - r2;
        float dg = g1 - g2;
        float db = b1 - b2;
        return WR * dr * dr + WG * dg * dg + WB * db * db;
    };

    const size_t SAMPLE_MAX = 200000; // balance quality/speed
    const int MAX_ITERS = 12;         // a bit more for better convergence

    struct SColor { float r, g, b, w; };
    std::vector<SColor> samples; // sRGB in [0..255], w = alpha weight [0..1]
    samples.reserve(std::min(n, SAMPLE_MAX));

    // Fill alpha buffer and gather samples (skip fully transparent)
    size_t step = n > SAMPLE_MAX ? (n / SAMPLE_MAX) : 1;
    size_t idx = 0;
    for (size_t y = 0; y < h; y++) {
        Pixel *row = input->data[y];
        for (size_t x = 0; x < w; x++, idx++) {
            Pixel p = row[x];
            alpha[y * w + x] = p.A;
            if ((idx % step) == 0 && p.A > 0) {
                float wgt = (float)p.A / 255.0f; // emphasize visible colours
                samples.push_back(SColor{ (float)p.R, (float)p.G, (float)p.B, wgt });
            }
        }
    }

    if (samples.empty()) {
        // Fallback: single black color
        palette[0] = palette[1] = palette[2] = 0; palette[3] = 0xFF;
        *paletteSize = 1;
        memset(indices, 0, n);
        return;
    }

    size_t K = std::min((size_t)256, samples.size());
    struct CColor { float r, g, b; };
    std::vector<CColor> centers;
    centers.reserve(K);

    // Farthest-point initialization (deterministic, robust)
    centers.push_back(CColor{ samples[0].r, samples[0].g, samples[0].b });
    std::vector<float> minD2(samples.size(), 1e30f);
    for (size_t i = 0; i < samples.size(); i++) {
        minD2[i] = dist2(samples[i].r, samples[i].g, samples[i].b,
                         centers[0].r, centers[0].g, centers[0].b);
    }
    while (centers.size() < K) {
        size_t farIdx = 0;
        float farVal = -1.0f;
        for (size_t i = 0; i < samples.size(); i++) {
            float v = minD2[i] * samples[i].w; // weight by alpha visibility
            if (v > farVal) { farVal = v; farIdx = i; }
        }
        centers.push_back(CColor{ samples[farIdx].r, samples[farIdx].g, samples[farIdx].b });
        for (size_t i = 0; i < samples.size(); i++) {
            float d2 = dist2(samples[i].r, samples[i].g, samples[i].b,
                             centers.back().r, centers.back().g, centers.back().b);
            if (d2 < minD2[i]) minD2[i] = d2;
        }
    }

    std::vector<float> counts(K, 0.0f);
    std::vector<CColor> sums(K, CColor{0,0,0});
    std::vector<uint16_t> assign(samples.size(), 0);

    for (int it = 0; it < MAX_ITERS; it++) {
        std::fill(counts.begin(), counts.end(), 0.0f);
        for (size_t k = 0; k < K; k++) { sums[k].r = sums[k].g = sums[k].b = 0.0f; }

        // Assign
        for (size_t i = 0; i < samples.size(); i++) {
            const auto &p = samples[i];
            float bestD = 1e30f; size_t bestK = 0;
            for (size_t k = 0; k < K; k++) {
                float d = dist2(p.r, p.g, p.b, centers[k].r, centers[k].g, centers[k].b);
                if (d < bestD) { bestD = d; bestK = k; }
            }
            assign[i] = (uint16_t)bestK;
            float wgt = p.w;
            counts[bestK] += wgt;
            sums[bestK].r += p.r * wgt; sums[bestK].g += p.g * wgt; sums[bestK].b += p.b * wgt;
        }

        // Update
        for (size_t k = 0; k < K; k++) {
            if (counts[k] <= 1e-6f) {
                // Re-seed an empty center deterministically
                size_t s = (k * 9973) % samples.size();
                centers[k].r = samples[s].r; centers[k].g = samples[s].g; centers[k].b = samples[s].b;
            } else {
                float inv = 1.0f / counts[k];
                centers[k].r = sums[k].r * inv;
                centers[k].g = sums[k].g * inv;
                centers[k].b = sums[k].b * inv;
            }
        }
    }

    // Build palette (BGRA) from non-empty clusters
    size_t paletteCount = 0;
    std::vector<CColor> paletteRgb; paletteRgb.reserve(256);
    for (size_t k = 0; k < K && paletteCount < 256; k++) {
        if (counts[k] <= 1e-6f) continue;
        uint8_t r8 = (uint8_t)std::min(255.0f, std::max(0.0f, centers[k].r));
        uint8_t g8 = (uint8_t)std::min(255.0f, std::max(0.0f, centers[k].g));
        uint8_t b8 = (uint8_t)std::min(255.0f, std::max(0.0f, centers[k].b));
        size_t p = paletteCount * 4;
        palette[p + 0] = b8;
        palette[p + 1] = g8;
        palette[p + 2] = r8;
        palette[p + 3] = 0xFF;
        paletteRgb.push_back(CColor{ (float)r8, (float)g8, (float)b8 });
        paletteCount++;
    }
    if (paletteCount == 0) {
        palette[0]=palette[1]=palette[2]=0; palette[3]=0xFF; paletteCount = 1;
        paletteRgb.push_back(CColor{0,0,0});
    }

    // Build high-resolution LUT (64x64x64) to speed up nearest search in mapping
    static const int LUTN = 64; // 6 bits/channel
    const int LUT_SIZE = LUTN * LUTN * LUTN;
    std::vector<uint8_t> lut(LUT_SIZE);
    auto lutIndex = [](int r, int g, int b){ return (r<<12) | (g<<6) | b; };
    for (int rr = 0; rr < LUTN; rr++) {
        float r_srgb = (float)((rr<<2) | 2); // bucket midpoint in 0..255
        for (int gg = 0; gg < LUTN; gg++) {
            float g_srgb = (float)((gg<<2) | 2);
            for (int bb = 0; bb < LUTN; bb++) {
                float b_srgb = (float)((bb<<2) | 2);
                int best = 0; float bestD = 1e30f;
                for (size_t k = 0; k < paletteRgb.size(); k++) {
                    const auto &c = paletteRgb[k];
                    float d = dist2(r_srgb, g_srgb, b_srgb, c.r, c.g, c.b);
                    if (d < bestD) { bestD = d; best = (int)k; }
                }
                lut[lutIndex(rr,gg,bb)] = (uint8_t)best;
            }
        }
    }

    // Map all pixels with Floyd–Steinberg error diffusion (serpentine)
    std::vector<float> errR(w + 2, 0.0f), errG(w + 2, 0.0f), errB(w + 2, 0.0f);
    std::vector<float> nextErrR(w + 2, 0.0f), nextErrG(w + 2, 0.0f), nextErrB(w + 2, 0.0f);

    for (size_t y = 0; y < h; y++) {
        bool rtl = (y % 2) == 1; // serpentine
        if (!rtl) {
            for (size_t x = 0; x < w; x++) {
                Pixel p = input->data[y][x];
                float a = (float)p.A / 255.0f;
                float r = std::min(255.0f, std::max(0.0f, (float)p.R + errR[x + 1]));
                float g = std::min(255.0f, std::max(0.0f, (float)p.G + errG[x + 1]));
                float b = std::min(255.0f, std::max(0.0f, (float)p.B + errB[x + 1]));

                // Nearest via LUT
                int best = lut[lutIndex((int)r >> 2, (int)g >> 2, (int)b >> 2)];
                indices[y * w + x] = (uint8_t)best;

                const auto &qc = paletteRgb[best];
                float er = (r - qc.r) * a;
                float eg = (g - qc.g) * a;
                float eb = (b - qc.b) * a;

                // Diffuse to neighbors (left-to-right):
                // right (x+1): 7/16, next row: (x-1)=3/16, (x)=5/16, (x+1)=1/16
                errR[x + 2] += er * (7.0f / 16.0f); errG[x + 2] += eg * (7.0f / 16.0f); errB[x + 2] += eb * (7.0f / 16.0f);
                nextErrR[x + 0] += er * (3.0f / 16.0f); nextErrG[x + 0] += eg * (3.0f / 16.0f); nextErrB[x + 0] += eb * (3.0f / 16.0f);
                nextErrR[x + 1] += er * (5.0f / 16.0f); nextErrG[x + 1] += eg * (5.0f / 16.0f); nextErrB[x + 1] += eb * (5.0f / 16.0f);
                nextErrR[x + 2] += er * (1.0f / 16.0f); nextErrG[x + 2] += eg * (1.0f / 16.0f); nextErrB[x + 2] += eb * (1.0f / 16.0f);
            }
        } else {
            for (size_t xi = 0; xi < w; xi++) {
                size_t x = w - 1 - xi;
                Pixel p = input->data[y][x];
                float a = (float)p.A / 255.0f;
                float r = std::min(255.0f, std::max(0.0f, (float)p.R + errR[x + 1]));
                float g = std::min(255.0f, std::max(0.0f, (float)p.G + errG[x + 1]));
                float b = std::min(255.0f, std::max(0.0f, (float)p.B + errB[x + 1]));

                int best = lut[lutIndex((int)r >> 2, (int)g >> 2, (int)b >> 2)];
                indices[y * w + x] = (uint8_t)best;

                const auto &qc = paletteRgb[best];
                float er = (r - qc.r) * a;
                float eg = (g - qc.g) * a;
                float eb = (b - qc.b) * a;

                // Diffuse to neighbors (right-to-left):
                // left (x-1): 7/16, next row: (x+1)=3/16, (x)=5/16, (x-1)=1/16
                errR[x + 0] += er * (7.0f / 16.0f); errG[x + 0] += eg * (7.0f / 16.0f); errB[x + 0] += eb * (7.0f / 16.0f);
                nextErrR[x + 2] += er * (3.0f / 16.0f); nextErrG[x + 2] += eg * (3.0f / 16.0f); nextErrB[x + 2] += eb * (3.0f / 16.0f);
                nextErrR[x + 1] += er * (5.0f / 16.0f); nextErrG[x + 1] += eg * (5.0f / 16.0f); nextErrB[x + 1] += eb * (5.0f / 16.0f);
                nextErrR[x + 0] += er * (1.0f / 16.0f); nextErrG[x + 0] += eg * (1.0f / 16.0f); nextErrB[x + 0] += eb * (1.0f / 16.0f);
            }
        }

        // Advance to next row
        errR.swap(nextErrR); errG.swap(nextErrG); errB.swap(nextErrB);
        std::fill(nextErrR.begin(), nextErrR.end(), 0.0f);
        std::fill(nextErrG.begin(), nextErrG.end(), 0.0f);
        std::fill(nextErrB.begin(), nextErrB.end(), 0.0f);
    }

    *paletteSize = paletteCount;
}

DECODER_FN(Blp) {
    // Decoder is not required for your workflow (PNG -> BLP). Stub it.
    // Report unsupported to allow other decoders to try, if any.
    return FAIL;
}

ENCODER_FN(Blp) {
    // Encode PixelArray into BLP1 (palette with 8-bit alpha), mip0 up to mip15
    if (!input || !output) return FAIL;

    const size_t w = input->width;
    const size_t h = input->height;
    const size_t n = w * h;

    if (w == 0 || h == 0) return FAIL;

    // Allocate temporary buffers
    uint8_t *palette = (uint8_t *)malloc(256 * 4);
    uint8_t *indices = (uint8_t *)malloc(n);
    uint8_t *alpha = (uint8_t *)malloc(n);
    if (!palette || !indices || !alpha) {
        if (palette) free(palette);
        if (indices) free(indices);
        if (alpha) free(alpha);
        return FAIL;
    }

    size_t paletteSize = 0;
    quantize_kmeans(input, palette, &paletteSize, indices, alpha);

    // Build palette LUT once, reuse for all mip levels
    std::vector<RgbF> paletteRgb;
    std::vector<uint8_t> lut64;
    buildLut64FromPalette(palette, paletteSize, paletteRgb, lut64);

    // Plan mip sizes and compute total output size.
    // Important: for POT textures, WebGL/OpenGL require a complete mip chain down to 1x1 when using mipmapped minification.
    // mdx-m3-viewer switches to LINEAR_MIPMAP_LINEAR when the file reports >1 mip, so we must not truncate the chain.
    const size_t HEADER_SIZE = 156; // BLP1 header: 7*4 + 16*4 offsets + 16*4 sizes
    const size_t PALETTE_BYTES = 256 * 4;
    const int MAX_MIP_COUNT = 16;   // BLP header supports up to 16 mip levels

    uint32_t mipOffsets[16];
    uint32_t mipSizes[16];
    for (int i = 0; i < 16; i++) { mipOffsets[i] = 0; mipSizes[i] = 0; }

    size_t mw = w;
    size_t mh = h;
    size_t totalMipBytes = 0;
    int mipCount = 0;
    for (int level = 0; level < MAX_MIP_COUNT; level++) {
        const size_t pixels = mw * mh;
        const size_t bytes = pixels + pixels; // indices + alpha (8-bit each)
        totalMipBytes += bytes;
        mipCount++;
        // Stop once we reach 1x1 (avoid repeating identical 1x1 mips for small textures)
        if (mw == 1 && mh == 1) break;
        // Next level dims (ceil half, but clamp to 1)
        mw = std::max((size_t)1, (mw + 1) / 2);
        mh = std::max((size_t)1, (mh + 1) / 2);
    }

    // Assemble output buffer
    const size_t outSize = HEADER_SIZE + PALETTE_BYTES + totalMipBytes;
    uint8_t *out = (uint8_t *)malloc(outSize);
    if (!out) {
        free(palette); free(indices); free(alpha);
        return FAIL;
    }

    // Build BLP1 header (156 bytes)
    uint8_t header[HEADER_SIZE];
    memset(header, 0, sizeof(header));
    header[0] = 'B'; header[1] = 'L'; header[2] = 'P'; header[3] = '1';
    writeUInt32LE(header, 4, 1);           // content = 1 (palette/Direct)
    writeUInt32LE(header, 8, 8);           // alphaBits = 8
    writeUInt32LE(header, 12, (uint32_t)w);
    writeUInt32LE(header, 16, (uint32_t)h);
    writeUInt32LE(header, 20, 0);          // extra
    writeUInt32LE(header, 24, (mipCount > 1) ? 1u : 0u); // hasMipmaps

    // Palette block
    memcpy(out, header, HEADER_SIZE);
    memset(out + HEADER_SIZE, 0, PALETTE_BYTES);
    memcpy(out + HEADER_SIZE, palette, paletteSize * 4);

    // Write mip data sequentially and record offsets/sizes
    uint32_t writeOffset = (uint32_t)(HEADER_SIZE + PALETTE_BYTES);
    mw = w; mh = h;

    // mip0 (already computed by quantize_kmeans)
    {
        const size_t pixels = mw * mh;
        const uint32_t bytes = (uint32_t)(pixels + pixels);
        mipOffsets[0] = writeOffset;
        mipSizes[0] = bytes;
        memcpy(out + writeOffset, indices, pixels);
        memcpy(out + writeOffset + (uint32_t)pixels, alpha, pixels);
        writeOffset += bytes;
    }

    // mip1..: downsample + map using same palette (no dithering for stability)
    PixelArray prevOwned;
    prevOwned.data = NULL; prevOwned.width = prevOwned.height = 0; prevOwned.type = EMPTY;
    bool hasPrevOwned = false;
    const PixelArray *prev = input;

    for (int level = 1; level < mipCount; level++) {
        mw = std::max((size_t)1, (mw + 1) / 2);
        mh = std::max((size_t)1, (mh + 1) / 2);

        PixelArray cur;
        cur.data = NULL; cur.width = cur.height = 0; cur.type = EMPTY;
        if (cur.Malloc(mw, mh) != SUCCESS) {
            if (hasPrevOwned) prevOwned.Free();
            free(out);
            free(palette); free(indices); free(alpha);
            return FAIL;
        }

        if (downsample2x2SeparateAlpha(prev, &cur) != SUCCESS) {
            cur.Free();
            if (hasPrevOwned) prevOwned.Free();
            free(out);
            free(palette); free(indices); free(alpha);
            return FAIL;
        }

        const size_t pixels = mw * mh;
        const uint32_t bytes = (uint32_t)(pixels + pixels);
        mipOffsets[level] = writeOffset;
        mipSizes[level] = bytes;

        uint8_t *idxDst = out + writeOffset;
        uint8_t *aDst = out + writeOffset + (uint32_t)pixels;
        mapToPaletteNoDither(&cur, lut64, idxDst, aDst);
        writeOffset += bytes;

        // Advance chain
        if (hasPrevOwned) prevOwned.Free();
        prevOwned = cur;
        hasPrevOwned = true;
        prev = &prevOwned;
        cur.data = NULL; cur.width = cur.height = 0; cur.type = EMPTY;
    }

    if (hasPrevOwned) prevOwned.Free();

    // Fill header mip tables (16 entries)
    for (int i = 0; i < 16; i++) {
        writeUInt32LE(out, 28 + (size_t)i * 4, mipOffsets[i]);
        writeUInt32LE(out, 28 + 64 + (size_t)i * 4, mipSizes[i]);
    }

    // Assign to output
    output->data = out;
    output->length = outSize;
    output->position = (unsigned long)outSize;

    free(palette);
    free(indices);
    free(alpha);
    return SUCCESS;
}

#endif // HAVE_BLP


