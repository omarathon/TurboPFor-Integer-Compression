// Two-band lock-step FoR+PFor fused decoder for NDVI ops.
//
// Bitstream format: two independent p4nenc256v16-compatible streams (one per
// band), plus a separate uint16 anchor array per band (one anchor per 256-block).
//
// Shared-b invariant: the encoder picks shared_b = max(opt_bA, opt_bB) per block
// so both bands' ctrl bytes carry the same b. This collapses the dispatch to 17
// entries instead of 17×17. M_CONST is suppressed in the encoder (all-equal
// residual blocks fall to PLAIN b=0) to maintain the invariant.
//
// Stage 1: noop (xor-sink) + add (widen-sum) ops, full pshufb exception merge.

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <utility>

#include "simdbitpacking_u16.h"   // simdpack_u16 (encode) — has extern "C" guards
extern "C" {
#include "ic.h"                   // bitpack16 / bitunpack16 / vbenc16 / vbdec16
}

#define BLK 256
enum { M_PLAIN = 0, M_BITMAP = 1, M_VBYTE = 2, M_CONST = 3 };
enum { OP_NOOP = 0, OP_ADD = 1 };  // expanded to NDVI_DIV/COUNT in stage 2

// ── kShuffle16: 8-bit mask → 16-byte pshufb pattern for 8-lane half ──────────
// Copied verbatim from simdbitpacking_u16_decode_inl.h. Each entry maps an 8-bit
// mask (which of 8 lanes are exceptions) to the byte-level gather pattern that
// packs exception values from a left-dense array into their destination lanes.
// Lane j → bytes 2j, 2j+1; 255 = zero-fill (no exception for that lane).
static const unsigned char kShuffle16[256][16] = {
  {255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255},
  {0,1,255,255,255,255,255,255,255,255,255,255,255,255,255,255},
  {255,255,0,1,255,255,255,255,255,255,255,255,255,255,255,255},
  {0,1,2,3,255,255,255,255,255,255,255,255,255,255,255,255},
  {255,255,255,255,0,1,255,255,255,255,255,255,255,255,255,255},
  {0,1,255,255,2,3,255,255,255,255,255,255,255,255,255,255},
  {255,255,0,1,2,3,255,255,255,255,255,255,255,255,255,255},
  {0,1,2,3,4,5,255,255,255,255,255,255,255,255,255,255},
  {255,255,255,255,255,255,0,1,255,255,255,255,255,255,255,255},
  {0,1,255,255,255,255,2,3,255,255,255,255,255,255,255,255},
  {255,255,0,1,255,255,2,3,255,255,255,255,255,255,255,255},
  {0,1,2,3,255,255,4,5,255,255,255,255,255,255,255,255},
  {255,255,255,255,0,1,2,3,255,255,255,255,255,255,255,255},
  {0,1,255,255,2,3,4,5,255,255,255,255,255,255,255,255},
  {255,255,0,1,2,3,4,5,255,255,255,255,255,255,255,255},
  {0,1,2,3,4,5,6,7,255,255,255,255,255,255,255,255},
  {255,255,255,255,255,255,255,255,0,1,255,255,255,255,255,255},
  {0,1,255,255,255,255,255,255,2,3,255,255,255,255,255,255},
  {255,255,0,1,255,255,255,255,2,3,255,255,255,255,255,255},
  {0,1,2,3,255,255,255,255,4,5,255,255,255,255,255,255},
  {255,255,255,255,0,1,255,255,2,3,255,255,255,255,255,255},
  {0,1,255,255,2,3,255,255,4,5,255,255,255,255,255,255},
  {255,255,0,1,2,3,255,255,4,5,255,255,255,255,255,255},
  {0,1,2,3,4,5,255,255,6,7,255,255,255,255,255,255},
  {255,255,255,255,255,255,0,1,2,3,255,255,255,255,255,255},
  {0,1,255,255,255,255,2,3,4,5,255,255,255,255,255,255},
  {255,255,0,1,255,255,2,3,4,5,255,255,255,255,255,255},
  {0,1,2,3,255,255,4,5,6,7,255,255,255,255,255,255},
  {255,255,255,255,0,1,2,3,4,5,255,255,255,255,255,255},
  {0,1,255,255,2,3,4,5,6,7,255,255,255,255,255,255},
  {255,255,0,1,2,3,4,5,6,7,255,255,255,255,255,255},
  {0,1,2,3,4,5,6,7,8,9,255,255,255,255,255,255},
  {255,255,255,255,255,255,255,255,255,255,0,1,255,255,255,255},
  {0,1,255,255,255,255,255,255,255,255,2,3,255,255,255,255},
  {255,255,0,1,255,255,255,255,255,255,2,3,255,255,255,255},
  {0,1,2,3,255,255,255,255,255,255,4,5,255,255,255,255},
  {255,255,255,255,0,1,255,255,255,255,2,3,255,255,255,255},
  {0,1,255,255,2,3,255,255,255,255,4,5,255,255,255,255},
  {255,255,0,1,2,3,255,255,255,255,4,5,255,255,255,255},
  {0,1,2,3,4,5,255,255,255,255,6,7,255,255,255,255},
  {255,255,255,255,255,255,0,1,255,255,2,3,255,255,255,255},
  {0,1,255,255,255,255,2,3,255,255,4,5,255,255,255,255},
  {255,255,0,1,255,255,2,3,255,255,4,5,255,255,255,255},
  {0,1,2,3,255,255,4,5,255,255,6,7,255,255,255,255},
  {255,255,255,255,0,1,2,3,255,255,4,5,255,255,255,255},
  {0,1,255,255,2,3,4,5,255,255,6,7,255,255,255,255},
  {255,255,0,1,2,3,4,5,255,255,6,7,255,255,255,255},
  {0,1,2,3,4,5,6,7,255,255,8,9,255,255,255,255},
  {255,255,255,255,255,255,255,255,0,1,2,3,255,255,255,255},
  {0,1,255,255,255,255,255,255,2,3,4,5,255,255,255,255},
  {255,255,0,1,255,255,255,255,2,3,4,5,255,255,255,255},
  {0,1,2,3,255,255,255,255,4,5,6,7,255,255,255,255},
  {255,255,255,255,0,1,255,255,2,3,4,5,255,255,255,255},
  {0,1,255,255,2,3,255,255,4,5,6,7,255,255,255,255},
  {255,255,0,1,2,3,255,255,4,5,6,7,255,255,255,255},
  {0,1,2,3,4,5,255,255,6,7,8,9,255,255,255,255},
  {255,255,255,255,255,255,0,1,2,3,4,5,255,255,255,255},
  {0,1,255,255,255,255,2,3,4,5,6,7,255,255,255,255},
  {255,255,0,1,255,255,2,3,4,5,6,7,255,255,255,255},
  {0,1,2,3,255,255,4,5,6,7,8,9,255,255,255,255},
  {255,255,255,255,0,1,2,3,4,5,6,7,255,255,255,255},
  {0,1,255,255,2,3,4,5,6,7,8,9,255,255,255,255},
  {255,255,0,1,2,3,4,5,6,7,8,9,255,255,255,255},
  {0,1,2,3,4,5,6,7,8,9,10,11,255,255,255,255},
  {255,255,255,255,255,255,255,255,255,255,255,255,0,1,255,255},
  {0,1,255,255,255,255,255,255,255,255,255,255,2,3,255,255},
  {255,255,0,1,255,255,255,255,255,255,255,255,2,3,255,255},
  {0,1,2,3,255,255,255,255,255,255,255,255,4,5,255,255},
  {255,255,255,255,0,1,255,255,255,255,255,255,2,3,255,255},
  {0,1,255,255,2,3,255,255,255,255,255,255,4,5,255,255},
  {255,255,0,1,2,3,255,255,255,255,255,255,4,5,255,255},
  {0,1,2,3,4,5,255,255,255,255,255,255,6,7,255,255},
  {255,255,255,255,255,255,0,1,255,255,255,255,2,3,255,255},
  {0,1,255,255,255,255,2,3,255,255,255,255,4,5,255,255},
  {255,255,0,1,255,255,2,3,255,255,255,255,4,5,255,255},
  {0,1,2,3,255,255,4,5,255,255,255,255,6,7,255,255},
  {255,255,255,255,0,1,2,3,255,255,255,255,4,5,255,255},
  {0,1,255,255,2,3,4,5,255,255,255,255,6,7,255,255},
  {255,255,0,1,2,3,4,5,255,255,255,255,6,7,255,255},
  {0,1,2,3,4,5,6,7,255,255,255,255,8,9,255,255},
  {255,255,255,255,255,255,255,255,0,1,255,255,2,3,255,255},
  {0,1,255,255,255,255,255,255,2,3,255,255,4,5,255,255},
  {255,255,0,1,255,255,255,255,2,3,255,255,4,5,255,255},
  {0,1,2,3,255,255,255,255,4,5,255,255,6,7,255,255},
  {255,255,255,255,0,1,255,255,2,3,255,255,4,5,255,255},
  {0,1,255,255,2,3,255,255,4,5,255,255,6,7,255,255},
  {255,255,0,1,2,3,255,255,4,5,255,255,6,7,255,255},
  {0,1,2,3,4,5,255,255,6,7,255,255,8,9,255,255},
  {255,255,255,255,255,255,0,1,2,3,255,255,4,5,255,255},
  {0,1,255,255,255,255,2,3,4,5,255,255,6,7,255,255},
  {255,255,0,1,255,255,2,3,4,5,255,255,6,7,255,255},
  {0,1,2,3,255,255,4,5,6,7,255,255,8,9,255,255},
  {255,255,255,255,0,1,2,3,4,5,255,255,6,7,255,255},
  {0,1,255,255,2,3,4,5,6,7,255,255,8,9,255,255},
  {255,255,0,1,2,3,4,5,6,7,255,255,8,9,255,255},
  {0,1,2,3,4,5,6,7,8,9,255,255,10,11,255,255},
  {255,255,255,255,255,255,255,255,255,255,0,1,2,3,255,255},
  {0,1,255,255,255,255,255,255,255,255,2,3,4,5,255,255},
  {255,255,0,1,255,255,255,255,255,255,2,3,4,5,255,255},
  {0,1,2,3,255,255,255,255,255,255,4,5,6,7,255,255},
  {255,255,255,255,0,1,255,255,255,255,2,3,4,5,255,255},
  {0,1,255,255,2,3,255,255,255,255,4,5,6,7,255,255},
  {255,255,0,1,2,3,255,255,255,255,4,5,6,7,255,255},
  {0,1,2,3,4,5,255,255,255,255,6,7,8,9,255,255},
  {255,255,255,255,255,255,0,1,255,255,2,3,4,5,255,255},
  {0,1,255,255,255,255,2,3,255,255,4,5,6,7,255,255},
  {255,255,0,1,255,255,2,3,255,255,4,5,6,7,255,255},
  {0,1,2,3,255,255,4,5,255,255,6,7,8,9,255,255},
  {255,255,255,255,0,1,2,3,255,255,4,5,6,7,255,255},
  {0,1,255,255,2,3,4,5,255,255,6,7,8,9,255,255},
  {255,255,0,1,2,3,4,5,255,255,6,7,8,9,255,255},
  {0,1,2,3,4,5,6,7,255,255,8,9,10,11,255,255},
  {255,255,255,255,255,255,255,255,0,1,2,3,4,5,255,255},
  {0,1,255,255,255,255,255,255,2,3,4,5,6,7,255,255},
  {255,255,0,1,255,255,255,255,2,3,4,5,6,7,255,255},
  {0,1,2,3,255,255,255,255,4,5,6,7,8,9,255,255},
  {255,255,255,255,0,1,255,255,2,3,4,5,6,7,255,255},
  {0,1,255,255,2,3,255,255,4,5,6,7,8,9,255,255},
  {255,255,0,1,2,3,255,255,4,5,6,7,8,9,255,255},
  {0,1,2,3,4,5,255,255,6,7,8,9,10,11,255,255},
  {255,255,255,255,255,255,0,1,2,3,4,5,6,7,255,255},
  {0,1,255,255,255,255,2,3,4,5,6,7,8,9,255,255},
  {255,255,0,1,255,255,2,3,4,5,6,7,8,9,255,255},
  {0,1,2,3,255,255,4,5,6,7,8,9,10,11,255,255},
  {255,255,255,255,0,1,2,3,4,5,6,7,8,9,255,255},
  {0,1,255,255,2,3,4,5,6,7,8,9,10,11,255,255},
  {255,255,0,1,2,3,4,5,6,7,8,9,10,11,255,255},
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,255,255},
  {255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,1},
  {0,1,255,255,255,255,255,255,255,255,255,255,255,255,2,3},
  {255,255,0,1,255,255,255,255,255,255,255,255,255,255,2,3},
  {0,1,2,3,255,255,255,255,255,255,255,255,255,255,4,5},
  {255,255,255,255,0,1,255,255,255,255,255,255,255,255,2,3},
  {0,1,255,255,2,3,255,255,255,255,255,255,255,255,4,5},
  {255,255,0,1,2,3,255,255,255,255,255,255,255,255,4,5},
  {0,1,2,3,4,5,255,255,255,255,255,255,255,255,6,7},
  {255,255,255,255,255,255,0,1,255,255,255,255,255,255,2,3},
  {0,1,255,255,255,255,2,3,255,255,255,255,255,255,4,5},
  {255,255,0,1,255,255,2,3,255,255,255,255,255,255,4,5},
  {0,1,2,3,255,255,4,5,255,255,255,255,255,255,6,7},
  {255,255,255,255,0,1,2,3,255,255,255,255,255,255,4,5},
  {0,1,255,255,2,3,4,5,255,255,255,255,255,255,6,7},
  {255,255,0,1,2,3,4,5,255,255,255,255,255,255,6,7},
  {0,1,2,3,4,5,6,7,255,255,255,255,255,255,8,9},
  {255,255,255,255,255,255,255,255,0,1,255,255,255,255,2,3},
  {0,1,255,255,255,255,255,255,2,3,255,255,255,255,4,5},
  {255,255,0,1,255,255,255,255,2,3,255,255,255,255,4,5},
  {0,1,2,3,255,255,255,255,4,5,255,255,255,255,6,7},
  {255,255,255,255,0,1,255,255,2,3,255,255,255,255,4,5},
  {0,1,255,255,2,3,255,255,4,5,255,255,255,255,6,7},
  {255,255,0,1,2,3,255,255,4,5,255,255,255,255,6,7},
  {0,1,2,3,4,5,255,255,6,7,255,255,255,255,8,9},
  {255,255,255,255,255,255,0,1,2,3,255,255,255,255,4,5},
  {0,1,255,255,255,255,2,3,4,5,255,255,255,255,6,7},
  {255,255,0,1,255,255,2,3,4,5,255,255,255,255,6,7},
  {0,1,2,3,255,255,4,5,6,7,255,255,255,255,8,9},
  {255,255,255,255,0,1,2,3,4,5,255,255,255,255,6,7},
  {0,1,255,255,2,3,4,5,6,7,255,255,255,255,8,9},
  {255,255,0,1,2,3,4,5,6,7,255,255,255,255,8,9},
  {0,1,2,3,4,5,6,7,8,9,255,255,255,255,10,11},
  {255,255,255,255,255,255,255,255,255,255,0,1,255,255,2,3},
  {0,1,255,255,255,255,255,255,255,255,2,3,255,255,4,5},
  {255,255,0,1,255,255,255,255,255,255,2,3,255,255,4,5},
  {0,1,2,3,255,255,255,255,255,255,4,5,255,255,6,7},
  {255,255,255,255,0,1,255,255,255,255,2,3,255,255,4,5},
  {0,1,255,255,2,3,255,255,255,255,4,5,255,255,6,7},
  {255,255,0,1,2,3,255,255,255,255,4,5,255,255,6,7},
  {0,1,2,3,4,5,255,255,255,255,6,7,255,255,8,9},
  {255,255,255,255,255,255,0,1,255,255,2,3,255,255,4,5},
  {0,1,255,255,255,255,2,3,255,255,4,5,255,255,6,7},
  {255,255,0,1,255,255,2,3,255,255,4,5,255,255,6,7},
  {0,1,2,3,255,255,4,5,255,255,6,7,255,255,8,9},
  {255,255,255,255,0,1,2,3,255,255,4,5,255,255,6,7},
  {0,1,255,255,2,3,4,5,255,255,6,7,255,255,8,9},
  {255,255,0,1,2,3,4,5,255,255,6,7,255,255,8,9},
  {0,1,2,3,4,5,6,7,255,255,8,9,255,255,10,11},
  {255,255,255,255,255,255,255,255,0,1,2,3,255,255,4,5},
  {0,1,255,255,255,255,255,255,2,3,4,5,255,255,6,7},
  {255,255,0,1,255,255,255,255,2,3,4,5,255,255,6,7},
  {0,1,2,3,255,255,255,255,4,5,6,7,255,255,8,9},
  {255,255,255,255,0,1,255,255,2,3,4,5,255,255,6,7},
  {0,1,255,255,2,3,255,255,4,5,6,7,255,255,8,9},
  {255,255,0,1,2,3,255,255,4,5,6,7,255,255,8,9},
  {0,1,2,3,4,5,255,255,6,7,8,9,255,255,10,11},
  {255,255,255,255,255,255,0,1,2,3,4,5,255,255,6,7},
  {0,1,255,255,255,255,2,3,4,5,6,7,255,255,8,9},
  {255,255,0,1,255,255,2,3,4,5,6,7,255,255,8,9},
  {0,1,2,3,255,255,4,5,6,7,8,9,255,255,10,11},
  {255,255,255,255,0,1,2,3,4,5,6,7,255,255,8,9},
  {0,1,255,255,2,3,4,5,6,7,8,9,255,255,10,11},
  {255,255,0,1,2,3,4,5,6,7,8,9,255,255,10,11},
  {0,1,2,3,4,5,6,7,8,9,10,11,255,255,12,13},
  {255,255,255,255,255,255,255,255,255,255,255,255,0,1,2,3},
  {0,1,255,255,255,255,255,255,255,255,255,255,2,3,4,5},
  {255,255,0,1,255,255,255,255,255,255,255,255,2,3,4,5},
  {0,1,2,3,255,255,255,255,255,255,255,255,4,5,6,7},
  {255,255,255,255,0,1,255,255,255,255,255,255,2,3,4,5},
  {0,1,255,255,2,3,255,255,255,255,255,255,4,5,6,7},
  {255,255,0,1,2,3,255,255,255,255,255,255,4,5,6,7},
  {0,1,2,3,4,5,255,255,255,255,255,255,6,7,8,9},
  {255,255,255,255,255,255,0,1,255,255,255,255,2,3,4,5},
  {0,1,255,255,255,255,2,3,255,255,255,255,4,5,6,7},
  {255,255,0,1,255,255,2,3,255,255,255,255,4,5,6,7},
  {0,1,2,3,255,255,4,5,255,255,255,255,6,7,8,9},
  {255,255,255,255,0,1,2,3,255,255,255,255,4,5,6,7},
  {0,1,255,255,2,3,4,5,255,255,255,255,6,7,8,9},
  {255,255,0,1,2,3,4,5,255,255,255,255,6,7,8,9},
  {0,1,2,3,4,5,6,7,255,255,255,255,8,9,10,11},
  {255,255,255,255,255,255,255,255,0,1,255,255,2,3,4,5},
  {0,1,255,255,255,255,255,255,2,3,255,255,4,5,6,7},
  {255,255,0,1,255,255,255,255,2,3,255,255,4,5,6,7},
  {0,1,2,3,255,255,255,255,4,5,255,255,6,7,8,9},
  {255,255,255,255,0,1,255,255,2,3,255,255,4,5,6,7},
  {0,1,255,255,2,3,255,255,4,5,255,255,6,7,8,9},
  {255,255,0,1,2,3,255,255,4,5,255,255,6,7,8,9},
  {0,1,2,3,4,5,255,255,6,7,255,255,8,9,10,11},
  {255,255,255,255,255,255,0,1,2,3,255,255,4,5,6,7},
  {0,1,255,255,255,255,2,3,4,5,255,255,6,7,8,9},
  {255,255,0,1,255,255,2,3,4,5,255,255,6,7,8,9},
  {0,1,2,3,255,255,4,5,6,7,255,255,8,9,10,11},
  {255,255,255,255,0,1,2,3,4,5,255,255,6,7,8,9},
  {0,1,255,255,2,3,4,5,6,7,255,255,8,9,10,11},
  {255,255,0,1,2,3,4,5,6,7,255,255,8,9,10,11},
  {0,1,2,3,4,5,6,7,8,9,255,255,10,11,12,13},
  {255,255,255,255,255,255,255,255,255,255,0,1,2,3,4,5},
  {0,1,255,255,255,255,255,255,255,255,2,3,4,5,6,7},
  {255,255,0,1,255,255,255,255,255,255,2,3,4,5,6,7},
  {0,1,2,3,255,255,255,255,255,255,4,5,6,7,8,9},
  {255,255,255,255,0,1,255,255,255,255,2,3,4,5,6,7},
  {0,1,255,255,2,3,255,255,255,255,4,5,6,7,8,9},
  {255,255,0,1,2,3,255,255,255,255,4,5,6,7,8,9},
  {0,1,2,3,4,5,255,255,255,255,6,7,8,9,10,11},
  {255,255,255,255,255,255,0,1,255,255,2,3,4,5,6,7},
  {0,1,255,255,255,255,2,3,255,255,4,5,6,7,8,9},
  {255,255,0,1,255,255,2,3,255,255,4,5,6,7,8,9},
  {0,1,2,3,255,255,4,5,255,255,6,7,8,9,10,11},
  {255,255,255,255,0,1,2,3,255,255,4,5,6,7,8,9},
  {0,1,255,255,2,3,4,5,255,255,6,7,8,9,10,11},
  {255,255,0,1,2,3,4,5,255,255,6,7,8,9,10,11},
  {0,1,2,3,4,5,6,7,255,255,8,9,10,11,12,13},
  {255,255,255,255,255,255,255,255,0,1,2,3,4,5,6,7},
  {0,1,255,255,255,255,255,255,2,3,4,5,6,7,8,9},
  {255,255,0,1,255,255,255,255,2,3,4,5,6,7,8,9},
  {0,1,2,3,255,255,255,255,4,5,6,7,8,9,10,11},
  {255,255,255,255,0,1,255,255,2,3,4,5,6,7,8,9},
  {0,1,255,255,2,3,255,255,4,5,6,7,8,9,10,11},
  {255,255,0,1,2,3,255,255,4,5,6,7,8,9,10,11},
  {0,1,2,3,4,5,255,255,6,7,8,9,10,11,12,13},
  {255,255,255,255,255,255,0,1,2,3,4,5,6,7,8,9},
  {0,1,255,255,255,255,2,3,4,5,6,7,8,9,10,11},
  {255,255,0,1,255,255,2,3,4,5,6,7,8,9,10,11},
  {0,1,2,3,255,255,4,5,6,7,8,9,10,11,12,13},
  {255,255,255,255,0,1,2,3,4,5,6,7,8,9,10,11},
  {0,1,255,255,2,3,4,5,6,7,8,9,10,11,12,13},
  {255,255,0,1,2,3,4,5,6,7,8,9,10,11,12,13},
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
};

// ── bit extraction (compile-time B, J) ───────────────────────────────────────
// OutReg J of a 256-block packed at B bits per value: bits [J*B, J*B+B) of each
// 16-bit lane's stream, potentially straddling into the next 16-bit word.
namespace {

template <int B, int J>
static inline __m256i ex(const __m256i *in) {
  constexpr int o = J * B, w = o >> 4, s = o & 15;
  __m256i v = _mm256_srli_epi16(_mm256_loadu_si256(in + w), s);
  if constexpr (s + B > 16)
    v = _mm256_or_si256(v, _mm256_slli_epi16(_mm256_loadu_si256(in + w + 1), 16 - s));
  constexpr int m = (B >= 16) ? 0xFFFF : ((1 << B) - 1);
  return _mm256_and_si256(v, _mm256_set1_epi16((short)m));
}

// ── per-OutReg pshufb exception merge ────────────────────────────────────────
// Applies the exception correction for OutReg J into `v`: loads excess values
// from the left-dense exception array _pex, shifts by B bits, gathers via
// pshufb using the 16-bit bitmap word bm, ORs into v. Advances _pex by the
// number of exceptions in this OutReg (popcount(bm)).
//
// Two 8-lane halves share the same pshufb table (kShuffle16, 8-bit mask → 16B).
// B must be < 16 for the shift to be defined; for B==16 (no exceptions), this
// is never called (xn==0, bm==0 → add_epi16(v,0)=v, correct but wasteful).
#define PFOR_MERGE_J(v, _pex, bm, B_CT)                                            \
  do {                                                                               \
    unsigned _m = (bm);                                                              \
    __m256i _exc = _mm256_set_m128i(                                                 \
        _mm_loadu_si128((const __m128i *)((_pex) + _mm_popcnt_u32(_m & 0xFFu))),    \
        _mm_loadu_si128((const __m128i *)(_pex)));                                   \
    if constexpr ((B_CT) < 16)                                                       \
      _exc = _mm256_slli_epi16(_exc, (B_CT));                                        \
    else                                                                              \
      _exc = _mm256_setzero_si256();                                                 \
    __m256i _shuf = _mm256_set_m128i(                                                \
        _mm_loadu_si128((const __m128i *)kShuffle16[_m >> 8]),                       \
        _mm_loadu_si128((const __m128i *)kShuffle16[_m & 0xFFu]));                   \
    _exc = _mm256_shuffle_epi8(_exc, _shuf);                                         \
    (v) = _mm256_add_epi16((v), _exc);                                               \
    (_pex) += _mm_popcnt_u32(_m);                                                    \
  } while (0)

// ── per-OutReg aggregate ──────────────────────────────────────────────────────
template <int OP>
static inline void acc_op2(__m256i va, __m256i vb, __m256i &accx) {
  if constexpr (OP == OP_NOOP) {
    accx = _mm256_xor_si256(accx, _mm256_xor_si256(va, vb));
  }
  if constexpr (OP == OP_ADD) {
    const __m256i z = _mm256_setzero_si256();
    accx = _mm256_add_epi32(accx, _mm256_unpacklo_epi16(va, z));
    accx = _mm256_add_epi32(accx, _mm256_unpackhi_epi16(va, z));
    accx = _mm256_add_epi32(accx, _mm256_unpacklo_epi16(vb, z));
    accx = _mm256_add_epi32(accx, _mm256_unpackhi_epi16(vb, z));
  }
}

// ── PLAIN sub-block kernel (no exceptions) ────────────────────────────────────
// Both bands are PLAIN at shared bit-width B. Extract OutRegs with compile-time
// shifts, apply FoR anchor broadcast, aggregate.
template <int OP, int B>
__attribute__((noinline)) static void sub_plain2(
    const __m256i *lowA, const __m256i *lowB,
    uint16_t ancA, uint16_t ancB, __m256i *accx) {
  __m256i x = *accx;
  const __m256i bcA = _mm256_set1_epi16((short)ancA);
  const __m256i bcB = _mm256_set1_epi16((short)ancB);
  [&]<int... J>(std::integer_sequence<int, J...>) {
    (([&] {
      __m256i va = _mm256_add_epi16(ex<B, J>(lowA), bcA);
      __m256i vb = _mm256_add_epi16(ex<B, J>(lowB), bcB);
      acc_op2<OP>(va, vb, x);
    }()), ...);
  }(std::make_integer_sequence<int, 16>{});
  *accx = x;
}

// ── PFOR sub-block kernel (at least one band has exceptions) ──────────────────
// Applies pshufb exception merge for both bands (safe even when one band is
// PLAIN: its bm16 is all-zero, so merge produces zero correction).
template <int OP, int B>
__attribute__((noinline)) static void sub_pfor2(
    const __m256i *lowA, const uint16_t *exA, const uint16_t *bm16A,
    const __m256i *lowB, const uint16_t *exB, const uint16_t *bm16B,
    uint16_t ancA, uint16_t ancB, __m256i *accx) {
  __m256i x = *accx;
  const __m256i bcA = _mm256_set1_epi16((short)ancA);
  const __m256i bcB = _mm256_set1_epi16((short)ancB);
  const uint16_t *_pA = exA, *_pB = exB;
  [&]<int... J>(std::integer_sequence<int, J...>) {
    (([&] {
      __m256i va = ex<B, J>(lowA);
      PFOR_MERGE_J(va, _pA, bm16A[J], B);
      va = _mm256_add_epi16(va, bcA);
      __m256i vb = ex<B, J>(lowB);
      PFOR_MERGE_J(vb, _pB, bm16B[J], B);
      vb = _mm256_add_epi16(vb, bcB);
      acc_op2<OP>(va, vb, x);
    }()), ...);
  }(std::make_integer_sequence<int, 16>{});
  *accx = x;
}

// ── dispatch tables ───────────────────────────────────────────────────────────
using PlainFn = void (*)(const __m256i *, const __m256i *,
                          uint16_t, uint16_t, __m256i *);
using PforFn  = void (*)(const __m256i *, const uint16_t *, const uint16_t *,
                          const __m256i *, const uint16_t *, const uint16_t *,
                          uint16_t, uint16_t, __m256i *);

static PlainFn g_plain_tbl[2][17];  // [op][b]
static PforFn  g_pfor_tbl[2][17];

template <int OP, int B>
static void reg_b() {
  g_plain_tbl[OP][B] = &sub_plain2<OP, B>;
  g_pfor_tbl[OP][B]  = &sub_pfor2<OP, B>;
}
template <int OP>
static void reg_op() {
  []<int... B>(std::integer_sequence<int, B...>) {
    (reg_b<OP, B>(), ...);
  }(std::make_integer_sequence<int, 17>{});
}
struct Init {
  Init() {
    reg_op<OP_NOOP>();
    reg_op<OP_ADD>();
  }
} g_init;

// ── helpers ───────────────────────────────────────────────────────────────────
static inline unsigned bitlen16(unsigned x) {
  return x ? (unsigned)(32 - __builtin_clz(x)) : 0u;
}
static inline size_t pad8(size_t bits) { return (bits + 7) >> 3; }
static inline unsigned popcnt32(const unsigned char *bm) {
  unsigned xn = 0;
  for (int w = 0; w < 4; ++w) {
    uint64_t bits; memcpy(&bits, bm + (size_t)w * 8, 8);
    xn += (unsigned)__builtin_popcountll(bits);
  }
  return xn;
}
static inline uint16_t read_u16le(const uint8_t *p) {
  return (uint16_t)((unsigned)p[0] | ((unsigned)p[1] << 8));
}

}  // namespace

// ── encoder ──────────────────────────────────────────────────────────────────
// Encode a single 256-element residual block at a pre-chosen bit-width b.
// maxbits = bitlen(max(residuals)) — used to compute bxe for exceptions.
// M_CONST is intentionally suppressed: all-equal residuals fall through to PLAIN
// so both band ctrl bytes always carry the same b (shared-b invariant).
static unsigned char *encode_block_at_b(
    const uint16_t *residuals, unsigned b, unsigned maxbits,
    unsigned char *op) {
  const uint16_t mask = b ? (uint16_t)((1u << b) - 1u) : 0u;
  const unsigned xn_total = [&] {
    unsigned k = 0;
    for (int i = 0; i < BLK; ++i) k += (bitlen16(residuals[i]) > b);
    return k;
  }();

  if (xn_total == 0) {
    *op++ = (unsigned char)((M_PLAIN << 5) | (b & 0x1fu));
    if (b) {
      uint16_t masked[BLK]; __m256i packed[16];
      for (int i = 0; i < BLK; ++i) masked[i] = (uint16_t)(residuals[i] & mask);
      simdpack_u16(masked, packed, b);
      memcpy(op, packed, (size_t)b * 32u); op += (size_t)b * 32u;
    }
    return op;
  }

  const unsigned bxe = maxbits - b;
  unsigned char bitmap[32]; memset(bitmap, 0, 32);
  uint16_t excess[BLK], masked[BLK];
#ifndef PFOR_BYTE_EXC
  unsigned char posbuf[BLK], vbexc[BLK * 3];
#endif
  unsigned k = 0;
  for (int i = 0; i < BLK; ++i) {
    masked[i] = (uint16_t)(residuals[i] & mask);
    if (bitlen16(residuals[i]) > b) {
      bitmap[i >> 3] |= (unsigned char)(1u << (i & 7));
      excess[k] = (uint16_t)(residuals[i] >> b);
#ifndef PFOR_BYTE_EXC
      posbuf[k] = (unsigned char)i;
#endif
      ++k;
    }
  }
#ifdef PFOR_BYTE_EXC
  // Byte-aligned excess: raw uint8 (bxe<=8) or uint16 (bxe>8). Avoids bitpack16
  // on encode and bitunpack16 on decode; pshufb merge works unchanged since
  // ex[] values are identical. Force BITMAP only (no VBYTE in byte-exc mode).
  const int use_vbyte = 0;
#else
  unsigned char *vbend = vbenc16(excess, xn_total, vbexc);
  const size_t vbyte_excess_sz = (size_t)(vbend - vbexc);
  const size_t bitmap_sz = 1u + 32u + pad8((size_t)xn_total * bxe);
  const size_t vbyte_sz  = 1u + vbyte_excess_sz + xn_total;
  const int use_vbyte = (xn_total <= 255) && (vbyte_sz < bitmap_sz);
#endif

  __m256i packed[16];
  if (b) {
    simdpack_u16(masked, packed, b);
  }

  if (!use_vbyte) {
    *op++ = (unsigned char)((M_BITMAP << 5) | (b & 0x1fu));
    *op++ = (unsigned char)bxe;
    memcpy(op, bitmap, 32); op += 32;
#ifdef PFOR_BYTE_EXC
    if (bxe <= 8u) { for (unsigned t = 0; t < xn_total; ++t) *op++ = (unsigned char)excess[t]; }
    else           { memcpy(op, excess, (size_t)xn_total * 2u); op += (size_t)xn_total * 2u; }
#else
    op = bitpack16(excess, xn_total, op, bxe);
#endif
    if (b) { memcpy(op, packed, (size_t)b * 32u); op += (size_t)b * 32u; }
#ifndef PFOR_BYTE_EXC
  } else {
    *op++ = (unsigned char)((M_VBYTE << 5) | (b & 0x1fu));
    *op++ = (unsigned char)xn_total;
    if (b) { memcpy(op, packed, (size_t)b * 32u); op += (size_t)b * 32u; }
    memcpy(op, vbexc, vbyte_excess_sz); op += vbyte_excess_sz;
    memcpy(op, posbuf, xn_total); op += xn_total;
#else
  } else { /* unreachable: use_vbyte=0 when PFOR_BYTE_EXC */ __builtin_unreachable();
#endif
  }
  return op;
}

extern "C" void p4nenc256v16_for2band(
    const uint16_t *inA, const uint16_t *inB, size_t n,
    uint16_t *anchorsA, uint16_t *anchorsB,
    uint8_t *outA, size_t *sizeA,
    uint8_t *outB, size_t *sizeB) {
  uint8_t *opA = outA, *opB = outB;
  uint16_t resA[BLK], resB[BLK];

  const size_t full = n & ~(size_t)(BLK - 1);
  for (size_t base = 0; base < full; base += BLK) {
    const uint16_t *blkA = inA + base, *blkB = inB + base;

    // FoR anchors = per-block min
    uint16_t aA = blkA[0], aB = blkB[0];
    for (int i = 1; i < BLK; ++i) {
      if (blkA[i] < aA) aA = blkA[i];
      if (blkB[i] < aB) aB = blkB[i];
    }
    *anchorsA++ = aA;
    *anchorsB++ = aB;

    // Residuals and per-band max bit-widths
    unsigned uA = 0, uB = 0;
    for (int i = 0; i < BLK; ++i) {
      resA[i] = (uint16_t)(blkA[i] - aA);
      resB[i] = (uint16_t)(blkB[i] - aB);
      uA |= resA[i]; uB |= resB[i];
    }
    const unsigned maxbitsA = bitlen16(uA), maxbitsB = bitlen16(uB);

    // Cost-model b selection per band (same logic as p4nenc256v16)
    auto best_b = [&](const uint16_t *res, unsigned maxbits) -> unsigned {
      unsigned cnt[18]; memset(cnt, 0, sizeof(cnt));
      for (int i = 0; i < BLK; ++i) cnt[bitlen16(res[i])]++;
      unsigned xn_b[17]; xn_b[16] = 0;
      for (int b = 15; b >= 0; --b) xn_b[b] = xn_b[b+1] + cnt[b+1];
      unsigned best = maxbits;
      size_t bcost = 1u + (size_t)32u * maxbits;
      for (int b = (int)maxbits - 1; b >= 0; --b) {
        unsigned xn = xn_b[b], bxe = maxbits - (unsigned)b;
        size_t bm = 1u + 32u + pad8((size_t)xn * bxe);
        unsigned vbe = bxe <= 7 ? 1u : (bxe <= 14 ? 2u : 3u);
        size_t vb = 1u + (size_t)xn * vbe + xn;
        size_t pos = bm < vb ? bm : vb;
        size_t cost = 1u + (size_t)32u * (unsigned)b + pos;
        if (cost < bcost) { bcost = cost; best = (unsigned)b; }
      }
      return best;
    };

    const unsigned bA = best_b(resA, maxbitsA);
    const unsigned bB = best_b(resB, maxbitsB);
    const unsigned b  = (bA > bB) ? bA : bB;  // shared b = max

    opA = encode_block_at_b(resA, b, maxbitsA, opA);
    opB = encode_block_at_b(resB, b, maxbitsB, opB);
  }

  // Raw tail + 32-byte over-read pad per band
  const size_t tail = n - full;
  if (tail) {
    memcpy(opA, inA + full, tail * sizeof(uint16_t)); opA += tail * sizeof(uint16_t);
    memcpy(opB, inB + full, tail * sizeof(uint16_t)); opB += tail * sizeof(uint16_t);
  }
  memset(opA, 0, 32); opA += 32;
  memset(opB, 0, 32); opB += 32;

  *sizeA = (size_t)(opA - outA);
  *sizeB = (size_t)(opB - outB);
}

// Bound on compressed size per band (same as p4nbound256v16_fused).
extern "C" size_t p4nbound256v16_for2band(size_t n) {
  return 2u * n + (n / BLK + 2u) * 64u + 256u;
}

// ── fused lockstep decoder ───────────────────────────────────────────────────
// Parses two independent PFor streams in lockstep, applies FoR anchors inline,
// aggregates per op — no decoded values written to memory.
extern "C" double ndvi2_pfor_for_indep(
    const uint8_t *encA, const uint16_t *anchorsA,
    const uint8_t *encB, const uint16_t *anchorsB,
    size_t n, int op) {
  static const uint16_t zero_bm[16] = {};
  __m256i accx = _mm256_setzero_si256();
  const uint8_t *ipA = encA, *ipB = encB;

  const size_t full = n & ~(size_t)(BLK - 1);
  for (size_t base = 0; base < full; base += BLK) {
    const uint16_t ancA = *anchorsA++, ancB = *anchorsB++;

    // Both streams are independent byte arrays — read both ctrl bytes first.
    const unsigned ctrlA = *ipA++, ctrlB = *ipB++;
    const unsigned modeA = (ctrlA >> 5) & 3u;
    const unsigned modeB = (ctrlB >> 5) & 3u;
    const unsigned b     = ctrlA & 0x1fu;   // shared b; encoder invariant: ctrlB & 0x1f == b

    // ── PLAIN fast path (no exception buffers on stack) ───────────────────
    if (modeA == M_PLAIN && modeB == M_PLAIN) {
      if (b == 0) {
        // All residuals zero; decoded value = anchor. Accumulate scalar.
        if (op == OP_ADD) {
          const int32_t contrib = ((int32_t)ancA + (int32_t)ancB) * (BLK / 8);
          accx = _mm256_add_epi32(accx, _mm256_set1_epi32(contrib));
        }
        continue;
      }
      const __m256i *lowA = (const __m256i *)ipA; ipA += (size_t)b * 32u;
      const __m256i *lowB = (const __m256i *)ipB; ipB += (size_t)b * 32u;
      g_plain_tbl[op][b](lowA, lowB, ancA, ancB, &accx);
      continue;
    }

    // ── Exception path — mirrors single-band FOR_BLOCKWALK inner scope ────
    // M_CONST is never emitted by encode_block_at_b (FoR residuals are never
    // all-equal since the anchor is the block min → at least one residual = 0).
    // Only PLAIN / BITMAP / VBYTE can appear here.
    {
      uint16_t exA[BLK + 16], exB[BLK + 16];
      const __m256i *lowA, *lowB;
      const uint16_t *bm16A = zero_bm, *bm16B = zero_bm;
      // xnA/xnB: exception counts — only used for PFOR_SKIP_MERGE_EXC scalar sum.
#ifdef PFOR_SKIP_MERGE_EXC
      unsigned xnA = 0, xnB = 0;
#endif

      // ── parse band A ──────────────────────────────────────────────────
      if (modeA == M_PLAIN) {
        lowA = (const __m256i *)ipA; ipA += (size_t)b * 32u;
      } else if (modeA == M_BITMAP) {
        const unsigned bxe = *ipA++;
        const uint8_t *bm = ipA; ipA += 32;
        const unsigned xn = popcnt32(bm);
#ifdef PFOR_SKIP_MERGE_EXC
        xnA = xn;
#endif
#ifdef PFOR_BYTE_EXC
        if (bxe <= 8u) { for (unsigned t=0; t<xn; ++t) exA[t] = ((const uint8_t*)ipA)[t]; ipA += xn; }
        else           { memcpy(exA, ipA, (size_t)xn*2u); ipA += (size_t)xn*2u; }
#else
        ipA = (const uint8_t *)bitunpack16(ipA, xn, exA, bxe);
#endif
        lowA = (const __m256i *)ipA; ipA += (size_t)b * 32u;
        bm16A = (const uint16_t *)bm;
#ifndef PFOR_BYTE_EXC
      } else {  // M_VBYTE (only reachable without PFOR_BYTE_EXC; encoder forces BITMAP under BYTE_EXC)
        const unsigned xn = *ipA++;
#ifdef PFOR_SKIP_MERGE_EXC
        xnA = xn;
#endif
        lowA = (const __m256i *)ipA; ipA += (size_t)b * 32u;
        ipA = (const uint8_t *)vbdec16((unsigned char *)ipA, xn, exA);
#ifdef PFOR_SKIP_MERGE_EXC
        ipA += xn;  // skip position bytes — not needed for scalar sum
#else
        { const uint8_t *pos = ipA; ipA += xn;
          unsigned char bmbufA[32]; memset(bmbufA, 0, 32);
          for (unsigned k = 0; k < xn; ++k)
            bmbufA[pos[k] >> 3] |= (unsigned char)(1u << (pos[k] & 7));
          bm16A = (const uint16_t *)bmbufA; }
#endif
#endif
      }

      // ── parse band B ──────────────────────────────────────────────────
      if (modeB == M_PLAIN) {
        lowB = (const __m256i *)ipB; ipB += (size_t)b * 32u;
      } else if (modeB == M_BITMAP) {
        const unsigned bxe = *ipB++;
        const uint8_t *bm = ipB; ipB += 32;
        const unsigned xn = popcnt32(bm);
#ifdef PFOR_SKIP_MERGE_EXC
        xnB = xn;
#endif
#ifdef PFOR_BYTE_EXC
        if (bxe <= 8u) { for (unsigned t=0; t<xn; ++t) exB[t] = ((const uint8_t*)ipB)[t]; ipB += xn; }
        else           { memcpy(exB, ipB, (size_t)xn*2u); ipB += (size_t)xn*2u; }
#else
        ipB = (const uint8_t *)bitunpack16(ipB, xn, exB, bxe);
#endif
        lowB = (const __m256i *)ipB; ipB += (size_t)b * 32u;
        bm16B = (const uint16_t *)bm;
#ifndef PFOR_BYTE_EXC
      } else {  // M_VBYTE (only reachable without PFOR_BYTE_EXC)
        const unsigned xn = *ipB++;
#ifdef PFOR_SKIP_MERGE_EXC
        xnB = xn;
#endif
        lowB = (const __m256i *)ipB; ipB += (size_t)b * 32u;
        ipB = (const uint8_t *)vbdec16((unsigned char *)ipB, xn, exB);
#ifdef PFOR_SKIP_MERGE_EXC
        ipB += xn;  // skip position bytes — not needed for scalar sum
#else
        { const uint8_t *pos = ipB; ipB += xn;
          unsigned char bmbufB[32]; memset(bmbufB, 0, 32);
          for (unsigned k = 0; k < xn; ++k)
            bmbufB[pos[k] >> 3] |= (unsigned char)(1u << (pos[k] & 7));
          bm16B = (const uint16_t *)bmbufB; }
#endif
#endif
      }

#ifdef PFOR_SKIP_MERGE_EXC
      // Skip-merge path: for OP_ADD, sum = plain_kernel(lowA+lowB) + sum(exA<<b) + sum(exB<<b).
      // Valid because sum(A+B) separates into per-band sums, each satisfying the same
      // sum(low) + sum(exc<<b) identity as single-band sum. Results (XOR-fold) will NOT
      // match the full-merge path — this is a timing diagnostic for the pshufb merge cost.
      if (op == OP_ADD) {
        g_plain_tbl[OP_ADD][b](lowA, lowB, ancA, ancB, &accx);
        int32_t exc_sum = 0;
        for (unsigned k = 0; k < xnA; ++k) exc_sum += (int32_t)exA[k] << b;
        for (unsigned k = 0; k < xnB; ++k) exc_sum += (int32_t)exB[k] << b;
        accx = _mm256_add_epi32(accx, _mm256_set_epi32(0,0,0,0,0,0,0,exc_sum));
      } else {
        g_pfor_tbl[op][b](lowA, exA, bm16A, lowB, exB, bm16B, ancA, ancB, &accx);
      }
#else
      g_pfor_tbl[op][b](lowA, exA, bm16A, lowB, exB, bm16B, ancA, ancB, &accx);
#endif
    }
  }

  // Raw tail (shared between bands at the same positions)
  const size_t tail = n - full;
  for (size_t i = 0; i < tail; ++i) {
    uint16_t va = read_u16le(ipA); ipA += 2;
    uint16_t vb = read_u16le(ipB); ipB += 2;
    if (op == OP_ADD)
      accx = _mm256_add_epi32(accx,
          _mm256_set_epi32(0,0,0,0,0,0,0,(int)((uint32_t)va + vb)));
  }

  // XOR-fold reduction: same formula as ndvi2_indep's xreduce() so that the
  // bench 'result' field matches simdcomp_fused for correctness cross-checking.
  alignas(32) int32_t v[8];
  _mm256_store_si256((__m256i*)v, accx);
  int64_t r = 0;
  for (int i = 0; i < 8; ++i) r ^= v[i];
  return (double)(r & 0xFFFF);
}
