#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>

#include "def.h"
#include "pri_queue.h"
#include "util.h"

namespace ip {

// -----------------------------------------------------------------------------
//  Dual_Greedy: a data structure for performing diversity-aware k-mips
// -----------------------------------------------------------------------------
class Dual_Greedy {
public:
    int   n_;                       // item cardinality
    int   d_;                       // dimensionality
    int   d2_;                      // dimensionality for 2nd space
    const float *item_set_;         // item set by MF
    const float *i2v_set_;          // i2v  set by item2vec
    
    int   *index_;                  // data index (descending order of mf_norm_)
    float *mf_norm_;                // l2-norm of items by MF
    
    // -------------------------------------------------------------------------
    Dual_Greedy(                    // constructor
        int   n,                        // item cardinality
        int   d,                        // dimensionality
        int   d2,                       // dimensionality for 2nd space
        const float *item_set,          // item set by MF
        const float *i2v_set);          // i2v  set by item2vec
    
    // -------------------------------------------------------------------------
    ~Dual_Greedy();                 // destructor
    
    // -------------------------------------------------------------------------
    int* dkmips_avg(                // diversity-aware k-mips (avg)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query);           // query (user) vector
    
    // -------------------------------------------------------------------------
    int* dkmips_plus_avg(           // diversity-aware k-mips (avg)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query);           // query (user) vector
    
    // -------------------------------------------------------------------------
    int* dkmips_max(                // diversity-aware k-mips (max)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query);           // query (user) vector
    
    // -------------------------------------------------------------------------
    int* dkmips_plus_max(           // diversity-aware k-mips (max)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query);           // query (user) vector
    
    // -------------------------------------------------------------------------
    //  the following four methods consider two spaces
    // -------------------------------------------------------------------------
    int* dkmips_avg_i2v(            // diversity-aware k-mips (avg)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query);           // query (user) vector
    
    // -------------------------------------------------------------------------
    int* dkmips_plus_avg_i2v(       // diversity-aware k-mips (avg)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query);           // query (user) vector
    
    // -------------------------------------------------------------------------
    int* dkmips_max_i2v(            // diversity-aware k-mips (max)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query);           // query (user) vector
    
    // -------------------------------------------------------------------------
    int* dkmips_plus_max_i2v(       // diversity-aware k-mips (max)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query);           // query (user) vector
    
    // -------------------------------------------------------------------------
    inline u64 get_estimated_memory() { // get estimated memory (bytes)
        u64 ret = sizeof(*this);
        ret += (sizeof(int) + sizeof(float))*n_; // index_ & mf_norm_
        return ret;
    }

protected:
    // -------------------------------------------------------------------------
    float calc_sum_ip(              // calc sum of ip in results
        int    id,                      // item id
        int    n,                       // number of results
        int    d,                       // item dimension from MF/item2vec
        Result *res,                    // results
        const float *items);            // items from MF/item2vec
        
    // -------------------------------------------------------------------------
    float calc_max_ip(              // calc max ip in results
        int    id,                      // item id
        int    n,                       // number of results
        int    d,                       // item dimension from MF/item2vec
        Result *res,                    // results
        const float *items);            // items from MF/item2vec
};

} // end namespace ip
