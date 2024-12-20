#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>

#include "def.h"
#include "pri_queue.h"
#include "util.h"
#include "bc_tree.h"

namespace ip {

// -----------------------------------------------------------------------------
//  BC_Greedy: a data structure for performing diversity-aware k-mips
// -----------------------------------------------------------------------------
class BC_Greedy {
public:
    int   n_;                       // item cardinality
    int   d_;                       // dimensionality
    int   d2_;                      // dimensionality for 2nd space
    const float *item_set_;         // item set by MF
    const float *i2v_set_;          // i2v  set by item2vec
    BC_Tree *tree_;                 // bc-tree for item set
    
    // -------------------------------------------------------------------------
    BC_Greedy(                      // constructor
        int   n,                        // item cardinality
        int   d,                        // dimensionality
        int   d2,                       // dimensionality for 2nd space
        const float *item_set,          // item set by MF
        const float *i2v_set);          // i2v  set by item2vec
    
    // -------------------------------------------------------------------------
    ~BC_Greedy();                   // destructor
    
    // -------------------------------------------------------------------------
    int* dkmips_avg(                // diversity-aware k-mips (avg)
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
    //  the following two methods consider two spaces
    // -------------------------------------------------------------------------
    int* dkmips_avg_i2v(            // diversity-aware k-mips (avg)
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
    inline u64 get_estimated_memory() { // get memory usage
        u64 ret = sizeof(*this);
        ret += tree_->get_estimated_memory();
        return ret;
    }

protected:
    // -------------------------------------------------------------------------
    void copy_item(                 // copy item with max id to result items
        int   max_id,                   // max id
        int   d,                        // item dimension from MF/item2vec
        const float* items,             // items from MF/item2vec
        float *dest);                   // destination of result items (return)
};

} // end namespace ip