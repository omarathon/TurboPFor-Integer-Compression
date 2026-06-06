// Fused-sum decode for TurboPFor 128v16 (uint16, SSE) — PROTOTYPE.
//
// Decodes a stock p4nenc128v16 stream but, instead of storing decoded values,
// accumulates a 32-bit-widened SIMD sum. The corrected value stream IS fully
// materialized in registers (real bitmap exception merge — no math shortcut);
// only the store stage is replaced with the same widen+accumulate FastPFor uses:
//   sum32 += unpacklo_epi16(corrected, 0);  sum32 += unpackhi_epi16(corrected, 0);
//
// GPLv2+ (derived from TurboPFor by powturbo). Modified for fused sum decode.

#include <stdint.h>
#include <immintrin.h>

#include "include_/conf.h"
#include "include_/vp4.h"
#include "include_/bitpack.h"
#include "include_/bitutil.h"
#include "include_/vint.h"
#include "include_/vlcbyte.h"
#include "include_/bitutil_.h"

#ifndef PAD8
#define PAD8(_x_) (((_x_)+8-1)/8)
#endif

// Exception left-pack shuffle table (defined in bitunpack.c, external linkage).
extern char _shuffle_16[256][16];

// Per-b vertical bitunpack dispatcher + extraction macros (BITUNPACK128V16_*).
#include "bitunpack_.h"

// Accumulate one corrected __m128i (8×u16) into a 4×u32 accumulator —
// identical widening to FastPFor's aggregate_sums_u16 (128-bit variant).
#define ACC_SUM16(psum, ovv) do { \
    *(psum) = _mm_add_epi32(*(psum), _mm_unpacklo_epi16((ovv), _mm_setzero_si128())); \
    *(psum) = _mm_add_epi32(*(psum), _mm_unpackhi_epi16((ovv), _mm_setzero_si128())); \
  } while (0)

// b==0 block macro (16 VOZ16 calls; VOZ16 is late-bound to each variant below).
#define BITUNBLK128V16_0(ip, _i_, _op_, _nb_, _parm_) { __m128i ov; \
  VOZ16(_op_, 0,ov,_nb_,_parm_); VOZ16(_op_, 1,ov,_nb_,_parm_); \
  VOZ16(_op_, 2,ov,_nb_,_parm_); VOZ16(_op_, 3,ov,_nb_,_parm_); \
  VOZ16(_op_, 4,ov,_nb_,_parm_); VOZ16(_op_, 5,ov,_nb_,_parm_); \
  VOZ16(_op_, 6,ov,_nb_,_parm_); VOZ16(_op_, 7,ov,_nb_,_parm_); \
  VOZ16(_op_, 8,ov,_nb_,_parm_); VOZ16(_op_, 9,ov,_nb_,_parm_); \
  VOZ16(_op_,10,ov,_nb_,_parm_); VOZ16(_op_,11,ov,_nb_,_parm_); \
  VOZ16(_op_,12,ov,_nb_,_parm_); VOZ16(_op_,13,ov,_nb_,_parm_); \
  VOZ16(_op_,14,ov,_nb_,_parm_); VOZ16(_op_,15,ov,_nb_,_parm_); }

// ── PFOR block WITHOUT exceptions: accumulate the unpacked low bits ───────────
#define BITMAX16 16
#define BITMAX32 32
#define BITUNPACK0(_parm_) _parm_ = _mm_setzero_si128()
#define VO16(op, i, ov, nb, parm)  ACC_SUM16(psum, ov)
#define VOZ16(op, i, ov, nb, parm) ACC_SUM16(psum, parm)

static const unsigned char *bitunpack128v16_sum(const unsigned char *in,
                                                unsigned b, __m128i *psum) {
  const unsigned char *ip = in + PAD8(128 * b);
  __m128i sv;
  BITUNPACK128V16(in, b, (unsigned short *)0, sv);
  return ip;
}

// ── PFOR block WITH exceptions: merge via bitmap left-pack, then accumulate ───
#undef  BITMAX16
#undef  BITMAX32
#undef  BITUNPACK0
#undef  VO16
#undef  VOZ16
#define BITMAX16 15
#define BITMAX32 31
#define BITUNPACK0(_parm_)
// b>=1: corrected = low_bits + (excess << b) at exception lanes.
#define VO16(op, i, ov, nb, parm)  m = *bb++; { \
    __m128i _c = _mm_add_epi16((ov), _mm_shuffle_epi8( \
        _mm_slli_epi16(_mm_loadu_si128((const __m128i *)pex), (nb)), \
        _mm_loadu_si128((const __m128i *)_shuffle_16[m]))); \
    ACC_SUM16(psum, _c); } pex += popcnt32(m)
// b==0: low bits are all 0, so corrected = excess (full value) at exception lanes.
#define VOZ16(op, i, ov, nb, parm) m = *bb++; { \
    __m128i _c = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)pex), \
        _mm_loadu_si128((const __m128i *)_shuffle_16[m])); \
    ACC_SUM16(psum, _c); } pex += popcnt32(m)

static const unsigned char *_bitunpack128v16_sum(const unsigned char *in,
                                                 unsigned b,
                                                 const uint16_t *pex,
                                                 const unsigned char *bb,
                                                 __m128i *psum) {
  const unsigned char *ip = in + PAD8(128 * b);
  unsigned m;
  __m128i sv;
  BITUNPACK128V16(in, b, (unsigned short *)0, sv);
  return ip;
}

// ── driver: mirrors P4NDEC's block loop, accumulating instead of storing ──────
// Returns the uint32 sum of all n decoded values. PROTOTYPE: requires n % 128 == 0
// and does not handle the variable-byte tail block.
#ifdef FUSED_PROFILE
#include <x86intrin.h>
uint64_t g_fused_kernel_cyc = 0;  // cycles inside the unpack+merge kernel
uint64_t g_fused_total_cyc  = 0;  // whole-decode cycles (driver = total - kernel)
#define KBEG() uint64_t _kt = __rdtsc()
#define KEND() (g_fused_kernel_cyc += __rdtsc() - _kt)
#else
#define KBEG() ((void)0)
#define KEND() ((void)0)
#endif

uint32_t p4ndec128v16_sum(const unsigned char *in, unsigned n) {
  const unsigned char *ip = in;
  __m128i sum = _mm_setzero_si128();
#ifdef FUSED_PROFILE
  uint64_t _tt = __rdtsc();
#endif

  for (unsigned blk = 0; blk < (n & ~127u); blk += 128) {
    unsigned b = *ip++, bx = 0;

    if ((b & 0xc0) == 0xc0) {                 // all values equal
      b &= 0x3f;
      uint16_t u = *(const uint16_t *)ip;
      if (b < 16) u = (uint16_t)(u & ((1u << b) - 1));
      __m128i bc = _mm_set1_epi16((short)u);
      { KBEG();
        for (int r = 0; r < 16; ++r) ACC_SUM16(&sum, bc);  // honest: sum the constant stream
        KEND(); }
      ip += (b + 7) / 8;
    } else if (!(b & 0x40)) {                 // PFOR
      if (b & 0x80) {                         // exceptions present
        bx = *ip++;
        b &= 0x7f;
        const unsigned char *bm = ip;         // 16-byte bitmap
        unsigned xn = popcnt64(ctou64(ip)) + popcnt64(ctou64(ip + 8));
        uint16_t ex[128 + 64];
        const unsigned char *ip2 = bitunpack16((unsigned char *)ip + 16, xn, ex, bx);
        { KBEG(); ip = _bitunpack128v16_sum(ip2, b, ex, bm, &sum); KEND(); }
      } else {                                // no exceptions
        { KBEG(); ip = bitunpack128v16_sum(ip, b, &sum); KEND(); }
      }
    } else {
      // Variable-byte hybrid exception block. It carries the same information as
      // a bitmap block (positions + excess), just serialized differently, and the
      // positions are sorted ascending. So rebuild the 16-byte bitmap from the
      // positions and reuse the register-only fused merge — no scatter, no output
      // buffer. Layout: [xn][low: PAD8(128*b)][vbenc(excess)][positions: xn bytes].
      b &= 0x3f;
      unsigned xn = *ip++;
      const unsigned char *low = ip;                          // low bits
      uint16_t ex[128 + 64];
      const unsigned char *pos =
          vbdec16((unsigned char *)(ip + PAD8(128 * b)), xn, ex);  // excess values
      uint8_t bm[16] = {0};
      for (unsigned i = 0; i < xn; ++i)                       // positions → bitmap
        bm[pos[i] >> 3] |= (uint8_t)(1u << (pos[i] & 7));
      { KBEG(); _bitunpack128v16_sum(low, b, ex, bm, &sum); KEND(); }  // register-only merge
      ip = pos + xn;
    }
  }

  // horizontal sum of the 4 u32 lanes
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
#ifdef FUSED_PROFILE
  g_fused_total_cyc += __rdtsc() - _tt;
#endif
  return (uint32_t)_mm_cvtsi128_si32(sum);
}
