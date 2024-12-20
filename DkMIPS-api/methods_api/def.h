#pragma once

#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include <stdint.h>

namespace ip {

// -----------------------------------------------------------------------------
//  macros
// -----------------------------------------------------------------------------
#define MIN(x, y)   (((x) < (y)) ? (x) : (y))
#define MAX(x, y)   (((x) > (y)) ? (x) : (y))
#define SQR(x)      ((x) * (x))
#define SUM(x, y)   ((x) + (y))
#define DIFF(x, y)  ((y) - (x))
#define SWAP(x, y)  { int tmp=x; x=y; y=tmp; }

// -----------------------------------------------------------------------------
//  typedef
// -----------------------------------------------------------------------------
typedef uint8_t  u08;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float    f32;
typedef double   f64;

// -----------------------------------------------------------------------------
//  general constants
// -----------------------------------------------------------------------------
const f32 E               = 2.7182818F;
const f32 PI              = 3.141592654F;
const f32 CHECK_ERROR     = 1e-9F;

const f32 MAXREAL         = 3.402823466e+38F;
const f32 MINREAL         = -MAXREAL;
const int MAXINT          = 2147483647;
const int MININT          = -MAXINT;

const int RANDOM_SEED     = 666;
const std::vector<int> Ks = { 10 }; // { 5,10,15,20,25 };

const int BLOCK_MAX_NUM   = 10000;  // H2_ALSH, SA_ALSH
const int N_PTS_INDEX     = 1000;   // H2_ALSH, SA_ALSH
const int CANDIDATES      = 100;    // SRP_LSH, QALSH
const int SCAN_SIZE       = 64;     // QALSH
const f32 APPRX_RATIO_NNS = 2.0f;   // Approximation Ratio for NNS  [1,+\infty)

const int LEAF_SIZE       = 100;    // BC_Tree & HBC_Tree
// const f32 INTERVAL_RATIO  = 0.5f;   // HBC_Tree
// const f32 BLOCK_NUM_RATIO = 0.05f;  // HBC_Tree

} // end namespace ip
