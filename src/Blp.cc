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
    // Encode PixelArray into BLP1 (palette with 8-bit alpha), mip0 only
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

    // Build BLP1 header (156 bytes)
    const size_t HEADER_SIZE = 156;
    uint8_t header[HEADER_SIZE];
    memset(header, 0, sizeof(header));
    header[0] = 'B'; header[1] = 'L'; header[2] = 'P'; header[3] = '1';
    writeUInt32LE(header, 4, 1);           // content = 1 (palette/Direct)
    writeUInt32LE(header, 8, 8);           // alphaBits = 8
    writeUInt32LE(header, 12, (uint32_t)w);
    writeUInt32LE(header, 16, (uint32_t)h);
    writeUInt32LE(header, 20, 0);          // extra
    writeUInt32LE(header, 24, 0);          // hasMipmaps = 0

    const uint32_t pixelDataOffset = (uint32_t)(HEADER_SIZE + 256 * 4);
    const uint32_t pixelDataSize = (uint32_t)(n + n); // indices + alpha
    writeUInt32LE(header, 28, pixelDataOffset); // mip0 offset
    writeUInt32LE(header, 28 + 64, pixelDataSize); // mip0 size

    // Assemble output buffer
    const size_t outSize = HEADER_SIZE + 256 * 4 + n + n;
    uint8_t *out = (uint8_t *)malloc(outSize);
    if (!out) {
        free(palette); free(indices); free(alpha);
        return FAIL;
    }

    memcpy(out, header, HEADER_SIZE);
    // Palette: if paletteSize < 256, remaining bytes are already zero-initialized
    memset(out + HEADER_SIZE, 0, 256 * 4);
    memcpy(out + HEADER_SIZE, palette, paletteSize * 4);
    memcpy(out + HEADER_SIZE + 256 * 4, indices, n);
    memcpy(out + HEADER_SIZE + 256 * 4 + n, alpha, n);

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


