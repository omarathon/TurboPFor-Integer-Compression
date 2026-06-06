// Fused-sum 256-width 16-bit PFor with Frame-of-Reference (FoR) correction.
//
// This is the TurboPFor counterpart of the simdcomp FoR-fused codecs
// (simdcomp_for_codec_uint16_w256.h). It reuses the SAME 256-element PFor
// bitstream as vp4d256v16_fused.c (low bits = simdcomp's AVX2 16-lane vertical
// bitpacking, exceptions = bit-packed excess + bitmap/vbyte positions), but:
//
//   * Encode operates on FoR RESIDUALS (value - per-window anchor), not raw
//     values. The anchor stream itself is held by the C++ wrapper (raw /
//     chunked-b-packed / hierarchical) — this file only encodes/decodes the
//     residual PFor payload.
//   * Decode adds the per-OutReg window anchor on the OutReg dependency chain,
//     right after the in-register exception merge and before the aggregate —
//     exactly the simdcomp FoR correction, just layered on top of PFor.
//       corrected = lowbits + (excess<<b) + anchor(window) ; aggregate
//     Anchor granularity (uniform / cscalar0-3 / half / quarter) and aggregate
//     (unpack-widen / madd) mirror the simdcomp FoR codec; selected at runtime
//     (loop-invariant) and dispatched to fully-inlined STATIC kernels so `sum`
//     stays in a YMM register across blocks (no cross-TU spill).
//
//   * CONST opcode is NOT emitted: a FoR residual block always contains a 0 in
//     every window (the anchor is that window's min), so it can never be
//     all-equal-nonzero. All-zero residual blocks become PLAIN b=0.
//
// GPLv2+ side wrapping the BSD simdcomp kernels + TurboPFor helpers.

#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#include "simdbitpacking_u16.h"  // simdpack_u16 (encode)
#include "ic.h"                  // bitpack16 / bitunpack16 / vbenc16 / vbdec16

// STATIC FoR-corrected decode kernels (no-exception corrected_* + PFOR-exception
// pfor_c*), all anchor modes × aggregates, compiled into THIS TU so they inline.
#include "simdbitpacking_u16_for_decode_inl.h"

#define BLK 256
enum { M_PLAIN = 0, M_BITMAP = 1, M_VBYTE = 2, M_CONST = 3 };
// FoR anchor granularity mode (matches simdcomp_for_w256_detail::decode_mode).
enum { FOR_UNIFORM = 0, FOR_SCALAR = 1, FOR_HALF = 2, FOR_QUARTER = 3 };

static inline unsigned bitlen16_for(unsigned x) {
  return x ? (unsigned)(32 - __builtin_clz(x)) : 0u;
}
static inline size_t pad8_for(size_t bits) { return (bits + 7) >> 3; }

// ── encoder (residuals → PFor payload; CONST disabled) ───────────────────────
// Byte-for-byte the same stream as p4nenc256v16 minus the CONST opcode.
size_t p4nenc256v16_for(uint16_t *in, size_t n, unsigned char *out) {
  unsigned char *op = out;
  uint16_t masked[BLK], excess[BLK];
  unsigned char posbuf[BLK], vbexc[BLK * 3], bitmap[32];
  __m256i packed[16];

  const size_t full = n & ~(size_t)(BLK - 1);
  for (size_t base = 0; base < full; base += BLK) {
    const uint16_t *blk = in + base;

    unsigned cnt[18];
    memset(cnt, 0, sizeof(cnt));
    unsigned u = 0;
    for (int i = 0; i < BLK; ++i) { unsigned v = blk[i]; u |= v; cnt[bitlen16_for(v)]++; }
    const unsigned maxbits = bitlen16_for(u);

    unsigned xn_b[17];
    xn_b[16] = 0;
    for (int b = 15; b >= 0; --b) xn_b[b] = xn_b[b + 1] + cnt[b + 1];

    unsigned best_b = maxbits;
    size_t best_cost = 1u + (size_t)32u * maxbits;
    for (int b = (int)maxbits - 1; b >= 0; --b) {
      unsigned xn = xn_b[b], bxe = maxbits - (unsigned)b;
      size_t bm = 1u + 32u + pad8_for((size_t)xn * bxe);
      unsigned vbe = bxe <= 7 ? 1u : (bxe <= 14 ? 2u : 3u);
      size_t vb = 1u + (size_t)xn * vbe + xn;
      size_t pos = bm < vb ? bm : vb;
      size_t cost = 1u + (size_t)32u * (unsigned)b + pos;
      if (cost < best_cost) { best_cost = cost; best_b = (unsigned)b; }
    }

    const unsigned b = best_b;
    const unsigned xn = xn_b[b];
    const uint16_t mask = b ? (uint16_t)((1u << b) - 1u) : 0u;

    if (xn == 0) {  // PLAIN (also covers the all-zero residual block at b=0)
      *op++ = (unsigned char)((M_PLAIN << 5) | (b & 0x1f));
      if (b) {
        for (int i = 0; i < BLK; ++i) masked[i] = (uint16_t)(blk[i] & mask);
        simdpack_u16(masked, packed, b);
        memcpy(op, packed, (size_t)b * 32u); op += (size_t)b * 32u;
      }
      continue;
    }

    const unsigned bxe = maxbits - b;
    memset(bitmap, 0, 32);
    unsigned k = 0;
    for (int i = 0; i < BLK; ++i) {
      uint16_t v = blk[i];
      masked[i] = (uint16_t)(v & mask);
      if (bitlen16_for(v) > b) {
        bitmap[i >> 3] |= (unsigned char)(1u << (i & 7));
        excess[k] = (uint16_t)(v >> b);
        posbuf[k] = (unsigned char)i;
        ++k;
      }
    }
    unsigned char *vbend = vbenc16(excess, xn, vbexc);
    const size_t vbyte_excess_sz = (size_t)(vbend - vbexc);
    const size_t bitmap_sz = 1u + 32u + pad8_for((size_t)xn * bxe);
    const size_t vbyte_sz  = 1u + vbyte_excess_sz + xn;
    const int use_vbyte = (xn <= 255) && (vbyte_sz < bitmap_sz);

    if (!use_vbyte) {  // BITMAP
      *op++ = (unsigned char)((M_BITMAP << 5) | (b & 0x1f));
      *op++ = (unsigned char)bxe;
      memcpy(op, bitmap, 32); op += 32;
      op = bitpack16(excess, xn, op, bxe);
      if (b) { simdpack_u16(masked, packed, b); memcpy(op, packed, (size_t)b * 32u); op += (size_t)b * 32u; }
    } else {           // VBYTE
      *op++ = (unsigned char)((M_VBYTE << 5) | (b & 0x1f));
      *op++ = (unsigned char)xn;
      if (b) { simdpack_u16(masked, packed, b); memcpy(op, packed, (size_t)b * 32u); op += (size_t)b * 32u; }
      memcpy(op, vbexc, vbyte_excess_sz); op += vbyte_excess_sz;
      memcpy(op, posbuf, xn); op += xn;
    }
  }

  const size_t tail = n - full;
  if (tail) { memcpy(op, in + full, tail * sizeof(uint16_t)); op += tail * sizeof(uint16_t); }
  memset(op, 0, 32); op += 32;
  return (size_t)(op - out);
}

// ── per-block FoR fused decode dispatch ──────────────────────────────────────
// `a_block` points at this block's first window anchor; for FOR_UNIFORM the
// block has a single anchor (a_block[0]). `mode`, `sh`, `madd` are loop-invariant
// (well-predicted). `is_exc` selects the cheap no-exception corrected kernel vs
// the PFOR-exception kernel. `scratch` is the unused `out` param (kernels are
// sum-only). NOTE: `b` ranges 0..16 for PLAIN, 0..15 for exception blocks.
static inline void for_block(const __m256i *low, unsigned b, int is_exc,
                             const uint16_t *ex, const uint16_t *bm16,
                             const uint16_t *a_block, uint16_t *scratch,
                             int mode, unsigned sh, int madd, __m256i *sum) {
  if (!is_exc) {  // ── PLAIN (no exceptions) ──
    if (mode == FOR_UNIFORM) {
      __m256i a = _mm256_set1_epi16((short)a_block[0]);
      if (madd) simdunpack_u16_il_corrected_uniform_madd(low, scratch, b, a, sum);
      else      simdunpack_u16_il_corrected_uniform(low, scratch, b, a, sum);
    } else if (mode == FOR_SCALAR) {
      if (madd) switch (sh) {
        case 4:  simdunpack_u16_il_cscalar0_madd(low, scratch, b, a_block, sum); break;
        case 5:  simdunpack_u16_il_cscalar1_madd(low, scratch, b, a_block, sum); break;
        case 6:  simdunpack_u16_il_cscalar2_madd(low, scratch, b, a_block, sum); break;
        default: simdunpack_u16_il_cscalar3_madd(low, scratch, b, a_block, sum); break;
      } else switch (sh) {
        case 4:  simdunpack_u16_il_cscalar0(low, scratch, b, a_block, sum); break;
        case 5:  simdunpack_u16_il_cscalar1(low, scratch, b, a_block, sum); break;
        case 6:  simdunpack_u16_il_cscalar2(low, scratch, b, a_block, sum); break;
        default: simdunpack_u16_il_cscalar3(low, scratch, b, a_block, sum); break;
      }
    } else if (mode == FOR_HALF) {
      if (madd) simdunpack_u16_il_corrected_half_madd(low, scratch, b, a_block, sum);
      else      simdunpack_u16_il_corrected_half(low, scratch, b, a_block, sum);
    } else {  // FOR_QUARTER
      if (madd) simdunpack_u16_il_corrected_quarter_madd(low, scratch, b, a_block, sum);
      else      simdunpack_u16_il_corrected_quarter(low, scratch, b, a_block, sum);
    }
  } else {  // ── PFOR exception block ──
    if (mode == FOR_UNIFORM) {
      __m256i a = _mm256_set1_epi16((short)a_block[0]);
      if (madd) simdunpack_u16_il_pfor_cuniform_madd(low, scratch, b, a, sum, ex, bm16);
      else      simdunpack_u16_il_pfor_cuniform(low, scratch, b, a, sum, ex, bm16);
    } else if (mode == FOR_SCALAR) {
      if (madd) switch (sh) {
        case 4:  simdunpack_u16_il_pfor_cscalar0_madd(low, scratch, b, a_block, sum, ex, bm16); break;
        case 5:  simdunpack_u16_il_pfor_cscalar1_madd(low, scratch, b, a_block, sum, ex, bm16); break;
        case 6:  simdunpack_u16_il_pfor_cscalar2_madd(low, scratch, b, a_block, sum, ex, bm16); break;
        default: simdunpack_u16_il_pfor_cscalar3_madd(low, scratch, b, a_block, sum, ex, bm16); break;
      } else switch (sh) {
        case 4:  simdunpack_u16_il_pfor_cscalar0(low, scratch, b, a_block, sum, ex, bm16); break;
        case 5:  simdunpack_u16_il_pfor_cscalar1(low, scratch, b, a_block, sum, ex, bm16); break;
        case 6:  simdunpack_u16_il_pfor_cscalar2(low, scratch, b, a_block, sum, ex, bm16); break;
        default: simdunpack_u16_il_pfor_cscalar3(low, scratch, b, a_block, sum, ex, bm16); break;
      }
    } else if (mode == FOR_HALF) {
      if (madd) simdunpack_u16_il_pfor_chalf_madd(low, scratch, b, a_block, sum, ex, bm16);
      else      simdunpack_u16_il_pfor_chalf(low, scratch, b, a_block, sum, ex, bm16);
    } else {  // FOR_QUARTER
      if (madd) simdunpack_u16_il_pfor_cquarter_madd(low, scratch, b, a_block, sum, ex, bm16);
      else      simdunpack_u16_il_pfor_cquarter(low, scratch, b, a_block, sum, ex, bm16);
    }
  }
}

// ── fused-sum FoR decoder ────────────────────────────────────────────────────
// `anchors` holds the materialized per-window anchors (the wrapper has already
// expanded raw/packed/hierarchical into this array). `w` is the FoR window,
// `sh` = log2(w), `mode` = FOR_* (chosen from w), `madd` selects the aggregate.
uint32_t p4ndec256v16_for_sum(const unsigned char *in, unsigned n,
                              const uint16_t *anchors, unsigned w, unsigned sh,
                              int mode, int madd) {
  (void)w;
  const unsigned char *ip = in;
  __m256i sum = _mm256_setzero_si256();
  static __thread uint16_t scratch[BLK];

  const unsigned full = n & ~(unsigned)(BLK - 1);
  unsigned base = 0;
  for (; base < full; base += BLK) {
    const unsigned ctrl = *ip++;
    const unsigned bmode = (ctrl >> 5) & 3u;
    const unsigned b = ctrl & 0x1f;
    const uint16_t *a_block = anchors + (base >> sh);

    if (bmode == M_PLAIN) {
      const __m256i *low = (const __m256i *)ip;
      ip += (size_t)b * 32u;
      // b==0 PLAIN still adds the per-window anchors honestly (corrected b==0).
      for_block(low, b, /*is_exc=*/0, NULL, NULL, a_block, scratch, mode, sh, madd, &sum);
      continue;
    }
    // M_CONST is never emitted for FoR residuals (see header); only exception
    // modes remain.
    uint16_t ex[BLK + 16];
    const __m256i *low;
    const uint16_t *bm16;
    unsigned char bmbuf[32];

    if (bmode == M_BITMAP) {
      const unsigned bxe = *ip++;
      const unsigned char *bm = ip; ip += 32;
      unsigned xn = 0;
      for (int wi = 0; wi < 4; ++wi) {
        uint64_t bits; memcpy(&bits, bm + (size_t)wi * 8, 8);
        xn += (unsigned)__builtin_popcountll(bits);
      }
      ip = bitunpack16(ip, xn, ex, bxe);
      low = (const __m256i *)ip; ip += (size_t)b * 32u;
      bm16 = (const uint16_t *)bm;
    } else {  // M_VBYTE
      const unsigned xn = *ip++;
      low = (const __m256i *)ip; ip += (size_t)b * 32u;
      ip = vbdec16((unsigned char *)ip, xn, ex);
      const unsigned char *pos = ip; ip += xn;
      memset(bmbuf, 0, 32);
      for (unsigned k = 0; k < xn; ++k)
        bmbuf[pos[k] >> 3] |= (unsigned char)(1u << (pos[k] & 7));
      bm16 = (const uint16_t *)bmbuf;
    }
    for_block(low, b, /*is_exc=*/1, ex, bm16, a_block, scratch, mode, sh, madd, &sum);
  }

  // Tail (n % 256): raw residuals + their per-window anchors, summed honestly.
  const unsigned tail = n - full;
  for (unsigned i = 0; i < tail; ++i) {
    uint16_t r = (uint16_t)((unsigned)ip[0] | ((unsigned)ip[1] << 8));
    ip += 2;
    unsigned elem = base + i;
    uint16_t v = (uint16_t)(r + anchors[elem >> sh]);
    sum = _mm256_add_epi32(sum, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, (int)v));
  }

  __m128i lo = _mm256_castsi256_si128(sum);
  __m128i hi = _mm256_extracti128_si256(sum, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return (uint32_t)_mm_cvtsi128_si32(s);
}
