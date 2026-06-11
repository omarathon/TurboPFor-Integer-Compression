// Fused-sum 256-width 16-bit PFor, built on simdcomp's AVX2 16-lane vertical
// bitpacking (simdpack_u16 / simdunpack_u16[_corrected]) for the low bits, with
// TurboPFor's exception coding for compression-ratio parity:
//   - excess values bit-packed at bx bits (bitpack16 / bitunpack16),
//   - exception positions stored as a 256-bit bitmap OR a TurboVByte hybrid
//     (vbenc16 excess + raw positions), whichever is smaller,
//   - a dedicated constant-block opcode.
//
// Decode is fused-sum only: for each 256-element block the corrected values are
// materialized in SIMD registers (low bits + per-OutReg exception correction
// added in-register) and 32-bit-widened into an accumulator; decoded values are
// never stored to memory.
//
// Element i within a block maps to OutReg v = i/16, lane l = i%16 (simdcomp's
// vertical layout loads consecutive 16-element chunks), so a per-element
// correction array corr[256] maps directly: corrections[v] = loadu(corr+16*v).
//
// Bitstream per 256-element block — ctrl byte = (mode<<5) | b, b in 0..16:
//   mode 0 PLAIN   [ctrl][lowbits:32*b]
//   mode 1 BITMAP  [ctrl][bx:1][bitmap:32][excess: bitpack16 @bx][lowbits:32*b]
//   mode 2 VBYTE   [ctrl][xn:1][lowbits:32*b][excess: vbenc16][positions: xn]
//   mode 3 CONST   [ctrl][value:2]
// Tail (n % 256): [raw: (n%256)*2]. The stream is padded with 32 zero bytes so
// the scalar bitunpack16 / vbdec16 over-reads stay in-bounds.
//
// GPLv2+ side (this file) wrapping the BSD simdcomp kernels + TurboPFor helpers.

#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#include "simdbitpacking_u16.h"  // simdpack_u16 (encode)
#include "ic.h"                  // bitpack16 / bitunpack16 / vbenc16 / vbdec16

// STATIC decode kernels compiled into THIS TU (fully inlined like TurboPFor-128's
// macros): simdunpack_u16_il (plain) + simdunpack_u16_pfor_il (PFOR merge).
#include "simdbitpacking_u16_decode_inl.h"

#define BLK 256
enum { M_PLAIN = 0, M_BITMAP = 1, M_VBYTE = 2, M_CONST = 3 };

static inline unsigned bitlen16(unsigned x) {
  return x ? (unsigned)(32 - __builtin_clz(x)) : 0u;
}
static inline size_t pad8(size_t bits) { return (bits + 7) >> 3; }

// ── encoder ──────────────────────────────────────────────────────────────────
size_t p4nenc256v16(uint16_t *in, size_t n, unsigned char *out) {
  unsigned char *op = out;
  uint16_t masked[BLK], excess[BLK];
  unsigned char posbuf[BLK], vbexc[BLK * 3], bitmap[32];
  __m256i packed[16];

  const size_t full = n & ~(size_t)(BLK - 1);
  for (size_t base = 0; base < full; base += BLK) {
    const uint16_t *blk = in + base;

    // Histogram of bit-lengths, OR of all values, equality to blk[0].
    unsigned cnt[18];
    memset(cnt, 0, sizeof(cnt));
    unsigned u = 0, a = blk[0], eq = 0;
    for (int i = 0; i < BLK; ++i) {
      unsigned v = blk[i];
      u |= v; eq += (v == a); cnt[bitlen16(v)]++;
    }
    const unsigned maxbits = bitlen16(u);

    // All-equal nonzero → constant block (all-zero falls through to PLAIN b=0).
    if (eq == BLK && a != 0) {
      *op++ = (unsigned char)(M_CONST << 5);
      *op++ = (unsigned char)(a & 0xff);
      *op++ = (unsigned char)(a >> 8);
      continue;
    }

    // xn_b[b] = #values needing > b bits = sum_{L>b} cnt[L].
    unsigned xn_b[17];
    xn_b[16] = 0;
    for (int b = 15; b >= 0; --b) xn_b[b] = xn_b[b + 1] + cnt[b + 1];

    // Bit-width selection: low bits 32*b, exceptions stored as the cheaper of a
    // bitmap (bit-packed excess) or a vbyte hybrid. Start from PLAIN at maxbits.
    unsigned best_b = maxbits;
    size_t best_cost = 1u + (size_t)32u * maxbits;  // ctrl + lowbits
    for (int b = (int)maxbits - 1; b >= 0; --b) {
      unsigned xn = xn_b[b], bxe = maxbits - (unsigned)b;
#ifdef PFOR_BYTE_EXC
      size_t bm = 1u + 32u + (size_t)xn * (bxe <= 8u ? 1u : 2u);  // byte-aligned excess
#else
      size_t bm = 1u + 32u + pad8((size_t)xn * bxe);          // bx + bitmap + excess
#endif
      unsigned vbe = bxe <= 7 ? 1u : (bxe <= 14 ? 2u : 3u);   // vbyte bytes/excess (est.)
      size_t vb = 1u + (size_t)xn * vbe + xn;                 // xn + excess + positions
      size_t pos = bm < vb ? bm : vb;
      size_t cost = 1u + (size_t)32u * (unsigned)b + pos;
      if (cost < best_cost) { best_cost = cost; best_b = (unsigned)b; }
    }

    const unsigned b = best_b;
    const unsigned xn = xn_b[b];
    const uint16_t mask = b ? (uint16_t)((1u << b) - 1u) : 0u;  // b<=16; b==16 has xn==0

    if (xn == 0) {  // ── PLAIN
      *op++ = (unsigned char)((M_PLAIN << 5) | (b & 0x1f));
      if (b) {
        for (int i = 0; i < BLK; ++i) masked[i] = (uint16_t)(blk[i] & mask);
        simdpack_u16(masked, packed, b);
        memcpy(op, packed, (size_t)b * 32u); op += (size_t)b * 32u;
      }
      continue;
    }

    // Collect low bits, excess (value>>b), positions, bitmap.
    const unsigned bxe = maxbits - b;
    memset(bitmap, 0, 32);
    unsigned k = 0;
    for (int i = 0; i < BLK; ++i) {
      uint16_t v = blk[i];
      masked[i] = (uint16_t)(v & mask);
      if (bitlen16(v) > b) {
        bitmap[i >> 3] |= (unsigned char)(1u << (i & 7));
        excess[k] = (uint16_t)(v >> b);
        posbuf[k] = (unsigned char)i;
        ++k;
      }
    }
    // Exact sizes for the two position encodings.
    unsigned char *vbend = vbenc16(excess, xn, vbexc);
    const size_t vbyte_excess_sz = (size_t)(vbend - vbexc);
    const size_t bitmap_sz = 1u + 32u + pad8((size_t)xn * bxe);
    const size_t vbyte_sz  = 1u + vbyte_excess_sz + xn;
#if defined(PFOR_BYTE_EXC) || defined(PFOR_SKIP_EXC)
    const int use_vbyte = 0;  // byte/skip paths only implement BITMAP
#else
    const int use_vbyte = (xn <= 255) && (vbyte_sz < bitmap_sz);
#endif

    if (!use_vbyte) {  // ── BITMAP
      *op++ = (unsigned char)((M_BITMAP << 5) | (b & 0x1f));
      *op++ = (unsigned char)bxe;
      memcpy(op, bitmap, 32); op += 32;
#ifdef PFOR_BYTE_EXC
      if (bxe <= 8u) { for (unsigned t = 0; t < xn; ++t) op[t] = (unsigned char)excess[t]; op += xn; }
      else           { memcpy(op, excess, (size_t)xn * 2u); op += (size_t)xn * 2u; }
#else
      op = bitpack16(excess, xn, op, bxe);
#endif
      if (b) {
        simdpack_u16(masked, packed, b);
        memcpy(op, packed, (size_t)b * 32u); op += (size_t)b * 32u;
      }
    } else {           // ── VBYTE
      *op++ = (unsigned char)((M_VBYTE << 5) | (b & 0x1f));
      *op++ = (unsigned char)xn;
      if (b) {
        simdpack_u16(masked, packed, b);
        memcpy(op, packed, (size_t)b * 32u); op += (size_t)b * 32u;
      }
      memcpy(op, vbexc, vbyte_excess_sz); op += vbyte_excess_sz;
      memcpy(op, posbuf, xn); op += xn;
    }
  }

  // Raw tail (n % 256), then 32 zero bytes so scalar over-reads stay in-bounds.
  const size_t tail = n - full;
  if (tail) { memcpy(op, in + full, tail * sizeof(uint16_t)); op += tail * sizeof(uint16_t); }
  memset(op, 0, 32); op += 32;
  return (size_t)(op - out);
}

// ── fused-sum decoder ─────────────────────────────────────────────────────────
#ifdef FUSED_PROFILE
#include <x86intrin.h>
uint64_t g_fused256_kernel_cyc = 0;  // cycles inside simdunpack* (the unpack kernel)
uint64_t g_fused256_total_cyc  = 0;  // cycles for the whole decode (driver = total-kernel)
#define KBEG() uint64_t _kt = __rdtsc()
#define KEND() (g_fused256_kernel_cyc += __rdtsc() - _kt)
#else
#define KBEG() ((void)0)
#define KEND() ((void)0)
#endif

uint32_t p4ndec256v16_sum(const unsigned char *in, unsigned n) {
  const unsigned char *ip = in;
  __m256i sum = _mm256_setzero_si256();
  static __thread uint16_t scratch[BLK];
#ifdef FUSED_PROFILE
  uint64_t _tt = __rdtsc();
#endif

  const unsigned full = n & ~(unsigned)(BLK - 1);
  for (unsigned base = 0; base < full; base += BLK) {
    const unsigned ctrl = *ip++;
    const unsigned mode = (ctrl >> 5) & 3u;
    const unsigned b = ctrl & 0x1f;

    if (mode == M_CONST) {
      uint16_t val = (uint16_t)((unsigned)ip[0] | ((unsigned)ip[1] << 8));
      ip += 2;
      // Sum 256 copies of val honestly (widen+accumulate 16 OutRegs of the
      // broadcast value), with NO memset of the scratch.
      { KBEG();
        __m256i bc = _mm256_set1_epi16((short)val), z = _mm256_setzero_si256();
        __m256i lo = _mm256_unpacklo_epi16(bc, z), hi = _mm256_unpackhi_epi16(bc, z);
        for (int r = 0; r < 16; ++r) {
          sum = _mm256_add_epi32(sum, lo);
          sum = _mm256_add_epi32(sum, hi);
        }
        KEND(); }
      continue;
    }
    if (mode == M_PLAIN) {
      const __m256i *low = (const __m256i *)ip;
      ip += (size_t)b * 32u;
      // b==0 is an all-zero block (xn==0): it adds 0 to the sum, so skip the
      // kernel entirely — the stock nullunpacker would needlessly memset 512 B
      // of scratch per block, which dominates on mostly-zero rasters.
      if (b) { KBEG(); simdunpack_u16_il(low, scratch, b, &sum); KEND(); }
      continue;
    }

    // Exception modes: decode excess + bitmap, then fused unpack + in-register
    // PFOR merge (simdunpack_u16_pfor) — no corr[] scatter, no corrections[].
    // ex[] has +16 u16 slack: the kernel over-reads up to 8 u16 per OutReg
    // (the shuffle zeroes the unused lanes, so the garbage never affects the sum).
    uint16_t ex[BLK + 16];
    const __m256i *low;
    const uint16_t *bm16;
    unsigned char bmbuf[32];

    if (mode == M_BITMAP) {
      const unsigned bxe = *ip++;
      const unsigned char *bm = ip; ip += 32;
      unsigned xn = 0;
      for (int w = 0; w < 4; ++w) {
        uint64_t bits; memcpy(&bits, bm + (size_t)w * 8, 8);
        xn += (unsigned)__builtin_popcountll(bits);
      }
#ifdef PFOR_BYTE_EXC
      if (bxe <= 8u) { for (unsigned t = 0; t < xn; ++t) ex[t] = ((const unsigned char *)ip)[t]; ip += xn; }
      else           { memcpy(ex, ip, (size_t)xn * 2u); ip += (size_t)xn * 2u; }
#else
      ip = bitunpack16(ip, xn, ex, bxe);   // excess, advances past packed bits
#endif
      low = (const __m256i *)ip; ip += (size_t)b * 32u;
      bm16 = (const uint16_t *)bm;          // 256-bit bitmap as 16 u16 words
    } else {  // M_VBYTE
      const unsigned xn = *ip++;
      low = (const __m256i *)ip; ip += (size_t)b * 32u;
      ip = vbdec16((unsigned char *)ip, xn, ex);  // excess, advances past vbyte
      const unsigned char *pos = ip; ip += xn;
      memset(bmbuf, 0, 32);                 // rebuild bitmap from positions
      for (unsigned k = 0; k < xn; ++k)
        bmbuf[pos[k] >> 3] |= (unsigned char)(1u << (pos[k] & 7));
      bm16 = (const uint16_t *)bmbuf;
    }
    { KBEG(); simdunpack_u16_pfor_il(low, scratch, b, &sum, ex, bm16); KEND(); }
  }

  // Tail: raw values, summed honestly (each materialized into lane 0).
  const unsigned tail = n - full;
  for (unsigned i = 0; i < tail; ++i) {
    uint16_t v = (uint16_t)((unsigned)ip[0] | ((unsigned)ip[1] << 8));
    ip += 2;
    sum = _mm256_add_epi32(sum, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, (int)v));
  }

  __m128i lo = _mm256_castsi256_si128(sum);
  __m128i hi = _mm256_extracti128_si256(sum, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
#ifdef FUSED_PROFILE
  g_fused256_total_cyc += __rdtsc() - _tt;
#endif
  return (uint32_t)_mm_cvtsi128_si32(s);
}

// madd-aggregate twin of p4ndec256v16_sum: identical bit-unpack + PFOR exception
// merge, but sums each OutReg with vpmaddwd (port 0/1) instead of unpack-widen
// (port 5). Valid only when every summed value < 2^15 — the TurboPFor-FoR nobc
// path calls this when the encoder's residual-madd-safe flag is set, so its
// residual decode aggregates the SAME way simdcomp's nobc_madd does (fair
// comparison). M_CONST never appears in FoR residuals (encoder disables it), so
// its honest widen loop here is dead — kept only for stream-format completeness.
uint32_t p4ndec256v16_sum_madd(const unsigned char *in, unsigned n) {
  const unsigned char *ip = in;
  __m256i sum = _mm256_setzero_si256();
  static __thread uint16_t scratch[BLK];

  const unsigned full = n & ~(unsigned)(BLK - 1);
  for (unsigned base = 0; base < full; base += BLK) {
    const unsigned ctrl = *ip++;
    const unsigned mode = (ctrl >> 5) & 3u;
    const unsigned b = ctrl & 0x1f;

    if (mode == M_CONST) {
      uint16_t val = (uint16_t)((unsigned)ip[0] | ((unsigned)ip[1] << 8));
      ip += 2;
      __m256i bc = _mm256_set1_epi16((short)val), z = _mm256_setzero_si256();
      __m256i lo = _mm256_unpacklo_epi16(bc, z), hi = _mm256_unpackhi_epi16(bc, z);
      for (int r = 0; r < 16; ++r) {
        sum = _mm256_add_epi32(sum, lo);
        sum = _mm256_add_epi32(sum, hi);
      }
      continue;
    }
    if (mode == M_PLAIN) {
      const __m256i *low = (const __m256i *)ip;
      ip += (size_t)b * 32u;
      // b==0 handled in the switch (no-op) — no driver-level guard, matching how
      // simdcomp's decode dispatches b==0 (fair comparison; no memset either way).
      simdunpack_u16_il_madd(low, scratch, b, &sum);
      continue;
    }

    uint16_t ex[BLK + 16];
    const __m256i *low;
    const uint16_t *bm16;
    unsigned char bmbuf[32];

    if (mode == M_BITMAP) {
      const unsigned bxe = *ip++;
      const unsigned char *bm = ip; ip += 32;
      unsigned xn = 0;
      for (int w = 0; w < 4; ++w) {
        uint64_t bits; memcpy(&bits, bm + (size_t)w * 8, 8);
        xn += (unsigned)__builtin_popcountll(bits);
      }
#if defined(PFOR_BYTE_EXC) && !defined(PFOR_SKIP_EXC)
      // Byte excess widened into ex[]; the pshufb merge below folds it into OutReg
      // positionally (general — works for min/max/NDVI/multiply, not just sum).
      if (bxe <= 8u) { for (unsigned t = 0; t < xn; ++t) ex[t] = ((const unsigned char *)ip)[t]; ip += xn; }
      else           { memcpy(ex, ip, (size_t)xn * 2u); ip += (size_t)xn * 2u; }
#elif !defined(PFOR_SKIP_EXC)
      ip = bitunpack16(ip, xn, ex, bxe);
#else
      ip += (((size_t)xn * bxe) + 7u) >> 3;  // skip excess; low-unpack only below
#endif
      low = (const __m256i *)ip; ip += (size_t)b * 32u;
      bm16 = (const uint16_t *)bm;
    } else {  // M_VBYTE  (encoder forces BITMAP under PFOR_SKIP_EXC; dead there)
      const unsigned xn = *ip++;
      low = (const __m256i *)ip; ip += (size_t)b * 32u;
      ip = vbdec16((unsigned char *)ip, xn, ex);
      const unsigned char *pos = ip; ip += xn;
      memset(bmbuf, 0, 32);
      for (unsigned k = 0; k < xn; ++k)
        bmbuf[pos[k] >> 3] |= (unsigned char)(1u << (pos[k] & 7));
      bm16 = (const uint16_t *)bmbuf;
    }
#ifndef PFOR_SKIP_EXC
    simdunpack_u16_pfor_il_madd(low, scratch, b, &sum, ex, bm16);
#else
    (void)bm16;  // skip the pshufb merge — unpack the low-bit plane only
    simdunpack_u16_il_madd(low, scratch, b, &sum);
#endif
  }

  const unsigned tail = n - full;
  for (unsigned i = 0; i < tail; ++i) {
    uint16_t v = (uint16_t)((unsigned)ip[0] | ((unsigned)ip[1] << 8));
    ip += 2;
    sum = _mm256_add_epi32(sum, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, (int)v));
  }

  __m128i lo = _mm256_castsi256_si128(sum);
  __m128i hi = _mm256_extracti128_si256(sum, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return (uint32_t)_mm_cvtsi128_si32(s);
}

// SIMD sum of `xn` raw uint8 (byte-aligned excess). vpsadbw against zero sums each
// 8-byte group into a 64-bit lane (one cheap op/32 bytes). The <32 remainder is a
// MASKED sad (not a scalar loop) — the scalar tail dominated at moderate density
// (xn<32 → every exception summed by hand). Over-reads up to 31 B into the next
// block / 32-byte end-pad (valid memory); masked lanes (>= r) contribute 0.
static inline uint64_t sum_bytes_u8(const unsigned char *p, unsigned xn) {
  __m256i acc = _mm256_setzero_si256();
  const __m256i z = _mm256_setzero_si256();
  unsigned k = 0;
  for (; k + 32 <= xn; k += 32)
    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_loadu_si256((const __m256i *)(p + k)), z));
  unsigned r = xn - k;  // 0..31 leftover
  if (r) {
    static const signed char kIota[32] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    __m256i v = _mm256_loadu_si256((const __m256i *)(p + k));
    __m256i mask = _mm256_cmpgt_epi8(_mm256_set1_epi8((char)r),
                                     _mm256_loadu_si256((const __m256i *)kIota));
    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_and_si256(v, mask), z));
  }
  return (uint64_t)_mm256_extract_epi64(acc, 0) + (uint64_t)_mm256_extract_epi64(acc, 1)
       + (uint64_t)_mm256_extract_epi64(acc, 2) + (uint64_t)_mm256_extract_epi64(acc, 3);
}

// SIMD sum of the first `xn` uint16 in ex[] (widen to u32). Bulk 16-at-a-time +
// scalar tail (never reads past ex[xn], so bitunpack16's slack stays untouched).
static inline uint64_t sum_excess_u16(const uint16_t *ex, unsigned xn) {
  __m256i acc = _mm256_setzero_si256();
  const __m256i z = _mm256_setzero_si256();
  unsigned k = 0;
  for (; k + 16 <= xn; k += 16) {
    __m256i v = _mm256_loadu_si256((const __m256i *)(ex + k));
    acc = _mm256_add_epi32(acc, _mm256_unpacklo_epi16(v, z));
    acc = _mm256_add_epi32(acc, _mm256_unpackhi_epi16(v, z));
  }
  __m128i lo = _mm256_castsi256_si128(acc);
  __m128i hi = _mm256_extracti128_si256(acc, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  uint64_t total = (uint32_t)_mm_cvtsi128_si32(s);
  for (; k < xn; ++k) total += ex[k];
  return total;
}

// SUM-ONLY fast decoder: exploits Σvalue = Σ(low bits) + (Σ excess) << b, so an
// exception block needs NO per-OutReg pshufb merge — decode its low bits like a
// PLAIN block (madd aggregate) and add the scalar (Σ excess)<<b. The bitmap /
// positions are parsed only to advance the stream, never consulted. Valid only
// for the SUM aggregate (position-agnostic) and when madd-safe (every b<=15, so
// the low bits are < 2^15). Used by the TurboPFor-FoR nobc path on madd-safe
// residuals — the exception merge was the dominant exception-block cost.
uint32_t p4ndec256v16_sum_fast(const unsigned char *in, unsigned n) {
  // NOTE: a fused scalar bit-cursor that sums the excess without the ex[]
  // round-trip was tried (FOR_EXC_NOFUSE) and was SLOWER — TurboPFor's
  // bitunpack16 is a heavily-unrolled scalar unpacker that beats a naive
  // per-value cursor, and the ex[] round-trip (L1, then SIMD reduce) is cheap.
  // So: bitunpack16 → ex[] → SIMD sum_excess_u16 stays.
  const unsigned char *ip = in;
  __m256i sum = _mm256_setzero_si256();      // low-bit (madd) sum
  uint64_t exc_acc = 0;                       // Σ over blocks of (Σ excess)<<b
  static __thread uint16_t scratch[BLK];
  static __thread uint16_t ex[BLK + 16];

  const unsigned full = n & ~(unsigned)(BLK - 1);
  for (unsigned base = 0; base < full; base += BLK) {
    const unsigned ctrl = *ip++;
    const unsigned mode = (ctrl >> 5) & 3u;
    const unsigned b = ctrl & 0x1f;

    if (mode == M_CONST) {  // never emitted for FoR residuals; honest fallback
      uint16_t val = (uint16_t)((unsigned)ip[0] | ((unsigned)ip[1] << 8));
      ip += 2;
      exc_acc += (uint64_t)BLK * val;
      continue;
    }
    if (mode == M_PLAIN) {
      const __m256i *low = (const __m256i *)ip;
      ip += (size_t)b * 32u;
      simdunpack_u16_il_madd(low, scratch, b, &sum);
      continue;
    }

    if (mode == M_BITMAP) {
      const unsigned bxe = *ip++;
      const unsigned char *bm = ip; ip += 32;
      unsigned xn = 0;
      for (int w = 0; w < 4; ++w) {
        uint64_t bits; memcpy(&bits, bm + (size_t)w * 8, 8);
        xn += (unsigned)__builtin_popcountll(bits);
      }
#if defined(PFOR_BYTE_EXC)
      // Byte-aligned excess: SIMD-sum the raw bytes directly (no bitunpack16).
      // With PFOR_SKIP_EXC *also* set: skip the byte-sum but keep the byte stream
      // — the FAIR floor for BYTE (same encoding, exceptions free).
  #ifndef PFOR_SKIP_EXC
      uint64_t es;
      if (bxe <= 8u) { es = sum_bytes_u8((const unsigned char *)ip, xn); ip += xn; }
      else           { es = sum_excess_u16((const uint16_t *)ip, xn); ip += (size_t)xn * 2u; }
      exc_acc += es << b;
  #else
      ip += (bxe <= 8u) ? xn : (size_t)xn * 2u;
  #endif
      const __m256i *low = (const __m256i *)ip; ip += (size_t)b * 32u;
      simdunpack_u16_il_madd(low, scratch, b, &sum);
#elif !defined(PFOR_SKIP_EXC)
      ip = bitunpack16(ip, xn, ex, bxe);
      const __m256i *low = (const __m256i *)ip; ip += (size_t)b * 32u;
      exc_acc += sum_excess_u16(ex, xn) << b;
      simdunpack_u16_il_madd(low, scratch, b, &sum);
#else
      // Diagnostic: skip the excess bit-extraction + reduce; just advance past
      // the packed excess and unpack the low-bit plane only (sum WRONG).
      ip += (((size_t)xn * bxe) + 7u) >> 3;
      const __m256i *low = (const __m256i *)ip; ip += (size_t)b * 32u;
      simdunpack_u16_il_madd(low, scratch, b, &sum);
#endif
    } else {  // M_VBYTE  (encoder forces BITMAP under PFOR_SKIP_EXC; dead there)
      const unsigned xn = *ip++;
      const __m256i *low = (const __m256i *)ip; ip += (size_t)b * 32u;
      ip = vbdec16((unsigned char *)ip, xn, ex);
      ip += xn;  // positions: skipped (irrelevant to the sum)
      exc_acc += sum_excess_u16(ex, xn) << b;
      simdunpack_u16_il_madd(low, scratch, b, &sum);
    }
  }

  const unsigned tail = n - full;
  for (unsigned i = 0; i < tail; ++i) {
    uint16_t v = (uint16_t)((unsigned)ip[0] | ((unsigned)ip[1] << 8));
    ip += 2;
    exc_acc += v;
  }

  __m128i lo = _mm256_castsi256_si128(sum);
  __m128i hi = _mm256_extracti128_si256(sum, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return (uint32_t)_mm_cvtsi128_si32(s) + (uint32_t)exc_acc;
}

// Unpack-widen twin of p4ndec256v16_sum_fast: identical factored exception sum
// (excess summed separately), but the low-bit plane is aggregated with the
// unpack-widen kernel (simdunpack_u16_il, port 5, unsigned-correct for any value)
// instead of madd. Used by the nobc+unpack path so the BYTE factored codec has a
// non-madd aggregate (for >2^15 residuals, and for fair madd-vs-unpack A/B).
uint32_t p4ndec256v16_sum_fast_unpack(const unsigned char *in, unsigned n) {
  const unsigned char *ip = in;
  __m256i sum = _mm256_setzero_si256();
  uint64_t exc_acc = 0;
  static __thread uint16_t scratch[BLK];
  static __thread uint16_t ex[BLK + 16];

  const unsigned full = n & ~(unsigned)(BLK - 1);
  for (unsigned base = 0; base < full; base += BLK) {
    const unsigned ctrl = *ip++;
    const unsigned mode = (ctrl >> 5) & 3u;
    const unsigned b = ctrl & 0x1f;

    if (mode == M_CONST) {
      uint16_t val = (uint16_t)((unsigned)ip[0] | ((unsigned)ip[1] << 8));
      ip += 2;
      exc_acc += (uint64_t)BLK * val;
      continue;
    }
    if (mode == M_PLAIN) {
      const __m256i *low = (const __m256i *)ip;
      ip += (size_t)b * 32u;
      simdunpack_u16_il(low, scratch, b, &sum);
      continue;
    }

    if (mode == M_BITMAP) {
      const unsigned bxe = *ip++;
      const unsigned char *bm = ip; ip += 32;
      unsigned xn = 0;
      for (int w = 0; w < 4; ++w) {
        uint64_t bits; memcpy(&bits, bm + (size_t)w * 8, 8);
        xn += (unsigned)__builtin_popcountll(bits);
      }
#if defined(PFOR_BYTE_EXC)
  #ifndef PFOR_SKIP_EXC
      uint64_t es;
      if (bxe <= 8u) { es = sum_bytes_u8((const unsigned char *)ip, xn); ip += xn; }
      else           { es = sum_excess_u16((const uint16_t *)ip, xn); ip += (size_t)xn * 2u; }
      exc_acc += es << b;
  #else
      ip += (bxe <= 8u) ? xn : (size_t)xn * 2u;
  #endif
      const __m256i *low = (const __m256i *)ip; ip += (size_t)b * 32u;
      simdunpack_u16_il(low, scratch, b, &sum);
#elif !defined(PFOR_SKIP_EXC)
      ip = bitunpack16(ip, xn, ex, bxe);
      const __m256i *low = (const __m256i *)ip; ip += (size_t)b * 32u;
      exc_acc += sum_excess_u16(ex, xn) << b;
      simdunpack_u16_il(low, scratch, b, &sum);
#else
      ip += (((size_t)xn * bxe) + 7u) >> 3;
      const __m256i *low = (const __m256i *)ip; ip += (size_t)b * 32u;
      simdunpack_u16_il(low, scratch, b, &sum);
#endif
    } else {  // M_VBYTE
      const unsigned xn = *ip++;
      const __m256i *low = (const __m256i *)ip; ip += (size_t)b * 32u;
      ip = vbdec16((unsigned char *)ip, xn, ex);
      ip += xn;
      exc_acc += sum_excess_u16(ex, xn) << b;
      simdunpack_u16_il(low, scratch, b, &sum);
    }
  }

  const unsigned tail = n - full;
  for (unsigned i = 0; i < tail; ++i) {
    uint16_t v = (uint16_t)((unsigned)ip[0] | ((unsigned)ip[1] << 8));
    ip += 2;
    exc_acc += v;
  }

  __m128i lo = _mm256_castsi256_si128(sum);
  __m128i hi = _mm256_extracti128_si256(sum, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return (uint32_t)_mm_cvtsi128_si32(s) + (uint32_t)exc_acc;
}

// Conservative compressed-size bound: no block exceeds PLAIN at b=16 (1 ctrl +
// 512 lowbits = 513), exception modes are only chosen when smaller. Tail <=
// (BLK-1)*2, plus the 32-byte over-read pad. 2*n covers the 512/block bulk.
size_t p4nbound256v16_fused(size_t n) {
  return 2u * n + (n / BLK + 2u) * 64u + 256u;
}
