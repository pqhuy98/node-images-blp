/*
 * Blp.cc - Minimal BLP1 (palette + 8-bit alpha) encoder/decoder stub
 */

#include "Image.h"

#ifdef HAVE_BLP

#include <stdlib.h>
#include <string.h>
#include <vector>
#include <unordered_map>
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

// Memoized nearest-palette lookup at 6-bit/channel precision. Entries are
// computed on demand at the bucket midpoint, which yields the exact same
// mapping as the previous eagerly-built 64^3 LUT, without paying the
// 64^3 * paletteSize brute-force cost (~67M distance evals) per texture.
struct LazyPaletteLut {
    std::vector<RgbF> palette;
    std::vector<uint8_t> lut;    // 64^3 entries
    std::vector<uint8_t> known;  // 64^3 computed flags

    void init(const std::vector<RgbF> &paletteRgb) {
        palette = paletteRgb;
        if (palette.empty()) palette.push_back(RgbF{0, 0, 0});
        static const int LUT_SIZE = 64 * 64 * 64;
        lut.assign(LUT_SIZE, 0);
        known.assign(LUT_SIZE, 0);
    }

    inline uint8_t lookup(int r8, int g8, int b8) {
        const int r6 = r8 >> 2;
        const int g6 = g8 >> 2;
        const int b6 = b8 >> 2;
        const int li = lutIndex64(r6, g6, b6);
        if (!known[li]) {
            const float r_srgb = (float)((r6 << 2) | 2); // bucket midpoint in 0..255
            const float g_srgb = (float)((g6 << 2) | 2);
            const float b_srgb = (float)((b6 << 2) | 2);
            int best = 0;
            float bestD = 1e30f;
            for (size_t k = 0; k < palette.size(); k++) {
                const auto &c = palette[k];
                float d = dist2_srgb(r_srgb, g_srgb, b_srgb, c.r, c.g, c.b);
                if (d < bestD) { bestD = d; best = (int)k; }
            }
            lut[li] = (uint8_t)best;
            known[li] = 1;
        }
        return lut[li];
    }
};

static void mapToPaletteNoDither(const PixelArray *input,
                                LazyPaletteLut &lut,
                                uint8_t *indices /* width*height */,
                                uint8_t *alpha /* width*height */) {
    const size_t w = input->width;
    const size_t h = input->height;
    for (size_t y = 0; y < h; y++) {
        Pixel *row = input->data[y];
        for (size_t x = 0; x < w; x++) {
            Pixel p = row[x];
            alpha[y * w + x] = p.A;
            indices[y * w + x] = lut.lookup(p.R, p.G, p.B);
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

// Quantization strategy:
// - Pass 1 scans all pixels once: fills the alpha plane, detects whether the
//   image already fits an exact <=256-color palette, and accumulates an
//   alpha-weighted 5-bit/channel histogram (32^3 buckets).
// - Exact path (<=256 unique colors, the common case for textures that were
//   palettized BLPs originally): the unique colors become the palette and
//   mip0 indices are an exact per-color mapping. No k-means, no dithering
//   (the representation is exact, so diffusion error would be zero anyway).
// - Otherwise: weighted k-means over the non-empty histogram bucket means
//   (typically a few thousand points instead of up to 200k pixel samples),
//   then Floyd-Steinberg dithering of mip0 via the memoized palette LUT.
static void quantize_kmeans(PixelArray *input,
                            uint8_t *palette /* 256*4 BGRA */, size_t *paletteSize,
                            uint8_t *indices /* width*height */,
                            uint8_t *alpha /* width*height */,
                            std::vector<RgbF> &paletteRgbOut,
                            LazyPaletteLut &lazyLut) {
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

    // --- Pass 1: alpha plane + exact-color map + histogram -----------------
    std::unordered_map<uint32_t, uint8_t> exactMap; // 24-bit RGB -> palette index
    exactMap.reserve(512);
    bool exactOk = true;

    static const int HIST_BITS = 5;
    static const int HIST_N = 1 << HIST_BITS;             // 32
    static const int HIST_SIZE = HIST_N * HIST_N * HIST_N; // 32768
    std::vector<float> histR(HIST_SIZE, 0.0f), histG(HIST_SIZE, 0.0f), histB(HIST_SIZE, 0.0f), histW(HIST_SIZE, 0.0f);
    bool anyVisible = false;

    for (size_t y = 0; y < h; y++) {
        Pixel *row = input->data[y];
        for (size_t x = 0; x < w; x++) {
            Pixel p = row[x];
            alpha[y * w + x] = p.A;

            if (exactOk) {
                const uint32_t key = ((uint32_t)p.R << 16) | ((uint32_t)p.G << 8) | (uint32_t)p.B;
                auto it = exactMap.find(key);
                if (it == exactMap.end()) {
                    if (exactMap.size() >= 256) {
                        exactOk = false;
                    } else {
                        exactMap.emplace(key, (uint8_t)exactMap.size());
                    }
                }
            }

            if (p.A > 0) {
                anyVisible = true;
                const float wgt = (float)p.A / 255.0f; // emphasize visible colours
                const int hi = (((int)p.R >> (8 - HIST_BITS)) << (2 * HIST_BITS))
                             | (((int)p.G >> (8 - HIST_BITS)) << HIST_BITS)
                             | ((int)p.B >> (8 - HIST_BITS));
                histR[hi] += (float)p.R * wgt;
                histG[hi] += (float)p.G * wgt;
                histB[hi] += (float)p.B * wgt;
                histW[hi] += wgt;
            }
        }
    }

    // --- Exact path ---------------------------------------------------------
    if (exactOk && !exactMap.empty()) {
        size_t count = exactMap.size();
        paletteRgbOut.clear();
        paletteRgbOut.resize(count);
        for (const auto &kv : exactMap) {
            const uint8_t r8 = (uint8_t)((kv.first >> 16) & 0xFF);
            const uint8_t g8 = (uint8_t)((kv.first >> 8) & 0xFF);
            const uint8_t b8 = (uint8_t)(kv.first & 0xFF);
            const size_t p = (size_t)kv.second * 4;
            palette[p + 0] = b8;
            palette[p + 1] = g8;
            palette[p + 2] = r8;
            palette[p + 3] = 0xFF;
            paletteRgbOut[kv.second] = RgbF{ (float)r8, (float)g8, (float)b8 };
        }
        *paletteSize = count;
        lazyLut.init(paletteRgbOut);

        for (size_t y = 0; y < h; y++) {
            Pixel *row = input->data[y];
            for (size_t x = 0; x < w; x++) {
                Pixel p = row[x];
                const uint32_t key = ((uint32_t)p.R << 16) | ((uint32_t)p.G << 8) | (uint32_t)p.B;
                indices[y * w + x] = exactMap.find(key)->second;
            }
        }
        return;
    }

    if (!anyVisible) {
        // Fallback: single black color
        palette[0] = palette[1] = palette[2] = 0; palette[3] = 0xFF;
        *paletteSize = 1;
        memset(indices, 0, n);
        paletteRgbOut.clear();
        paletteRgbOut.push_back(RgbF{0, 0, 0});
        lazyLut.init(paletteRgbOut);
        return;
    }

    // --- K-means over histogram bucket means --------------------------------
    struct SColor { float r, g, b, w; };
    std::vector<SColor> samples;
    samples.reserve(4096);
    for (int hi = 0; hi < HIST_SIZE; hi++) {
        const float wgt = histW[hi];
        if (wgt <= 0.0f) continue;
        const float inv = 1.0f / wgt;
        samples.push_back(SColor{ histR[hi] * inv, histG[hi] * inv, histB[hi] * inv, wgt });
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

    const int MAX_ITERS = 12;
    std::vector<float> counts(K, 0.0f);
    std::vector<CColor> sums(K, CColor{0,0,0});

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
            float wgt = p.w;
            counts[bestK] += wgt;
            sums[bestK].r += p.r * wgt; sums[bestK].g += p.g * wgt; sums[bestK].b += p.b * wgt;
        }

        // Update
        float maxShift2 = 0.0f;
        for (size_t k = 0; k < K; k++) {
            if (counts[k] <= 1e-6f) {
                // Re-seed an empty center deterministically
                size_t s = (k * 9973) % samples.size();
                centers[k].r = samples[s].r; centers[k].g = samples[s].g; centers[k].b = samples[s].b;
                maxShift2 = 1e30f;
            } else {
                float inv = 1.0f / counts[k];
                float nr = sums[k].r * inv;
                float ng = sums[k].g * inv;
                float nb = sums[k].b * inv;
                float shift2 = dist2(nr, ng, nb, centers[k].r, centers[k].g, centers[k].b);
                if (shift2 > maxShift2) maxShift2 = shift2;
                centers[k].r = nr;
                centers[k].g = ng;
                centers[k].b = nb;
            }
        }
        if (maxShift2 < 0.25f) break; // converged
    }

    // Build palette (BGRA) from non-empty clusters
    size_t paletteCount = 0;
    paletteRgbOut.clear();
    paletteRgbOut.reserve(256);
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
        paletteRgbOut.push_back(RgbF{ (float)r8, (float)g8, (float)b8 });
        paletteCount++;
    }
    if (paletteCount == 0) {
        palette[0]=palette[1]=palette[2]=0; palette[3]=0xFF; paletteCount = 1;
        paletteRgbOut.push_back(RgbF{0,0,0});
    }
    *paletteSize = paletteCount;

    lazyLut.init(paletteRgbOut);

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

                // Nearest via memoized LUT
                int best = lazyLut.lookup((int)r, (int)g, (int)b);
                indices[y * w + x] = (uint8_t)best;

                const auto &qc = paletteRgbOut[best];
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

                int best = lazyLut.lookup((int)r, (int)g, (int)b);
                indices[y * w + x] = (uint8_t)best;

                const auto &qc = paletteRgbOut[best];
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
    std::vector<RgbF> paletteRgb;
    LazyPaletteLut lut64;
    quantize_kmeans(input, palette, &paletteSize, indices, alpha, paletteRgb, lut64);

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
