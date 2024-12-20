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
//  Linear: a data structure for performing k-mips
// -----------------------------------------------------------------------------
class Linear {
public:
    int   n_;                       // item cardinality
    int   d_;                       // dimensionality
    int   d2_;                      // dimensionality for 2nd space
    const float *item_set_;         // item set by MF
    const float *i2v_set_;          // i2v  set by item2vec
    
    // -------------------------------------------------------------------------
    Linear(                         // constructor
        int   n,                        // item cardinality
        int   d,                        // dimensionality
        int   d2,                       // dimensionality for 2nd space
        const float *item_set,          // item set by MF
        const float *i2v_set);          // i2v  set by item2vec
    
    // -------------------------------------------------------------------------
    ~Linear();                      // destructor
    
    // -------------------------------------------------------------------------
    void kmips_avg(                 // k-mips (for avg)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query,            // query (user) vector
        float  &mmr,                    // mmr (return)
        float  &mean_k_ip,              // mean of k_ip (return)
        float  &mean_pair_ip,           // mean pair-wise ip (return)
        Result *res);                   // result set S (return)
    
    // -------------------------------------------------------------------------
    void kmips_max(                 // k-mips (for max)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query,            // query (user) vector
        float  &mmr,                    // mmr (return)
        float  &mean_k_ip,              // mean of k_ip (return)
        float  &max_pair_ip,            // max pair-wise ip (return)
        Result *res);                   // result set S (return)
    
    // -------------------------------------------------------------------------
    //  the following two methods consider two spaces
    // -------------------------------------------------------------------------
    void kmips_avg_i2v(             // k-mips (for avg)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query,            // query (user) vector
        float  &mmr,                    // mmr (return)
        float  &mean_k_ip,              // mean of k_ip (return)
        float  &mean_pair_ip,           // mean pair-wise ip (return)
        Result *res);                   // result set S (return)
    
    // -------------------------------------------------------------------------
    void kmips_max_i2v(             // k-mips (for max)
        int    k,                       // top-k value
        float  lambda,                  // balance factor
        float  c,                       // scale factor
        const  float *query,            // query (user) vector
        float  &mmr,                    // mmr (return)
        float  &mean_k_ip,              // mean of k_ip (return)
        float  &max_pair_ip,            // max pair-wise ip (return)
        Result *res);                   // result set S (return)
    
    // -------------------------------------------------------------------------
    inline u64 get_estimated_memory() { return sizeof(*this); }

protected:
    // -------------------------------------------------------------------------
    void linear_scan(               // linear scan all items
        const float *query,             // query (user) vector
        MaxK_List *list);               // top-k list (return)
};

} // end namespace ip
