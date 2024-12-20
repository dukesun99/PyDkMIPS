#include "linear.h"

namespace ip {

// -----------------------------------------------------------------------------
Linear::Linear(                     // constructor
    int   n,                            // item cardinality
    int   d,                            // dimensionality
    int   d2,                           // dimensionality for 2nd space
    const float *item_set,              // item set by MF
    const float *i2v_set)               // i2v  set by item2vec
    : n_(n), d_(d), d2_(d2), item_set_(item_set), i2v_set_(i2v_set)
{
}

// -----------------------------------------------------------------------------
Linear::~Linear()                   // destructor
{
}

// -----------------------------------------------------------------------------
void Linear::linear_scan(           // linear scan all items
    const float *query,                 // query (user) vector
    MaxK_List *list)                    // top-k list (return)
{
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        float ip = calc_inner_product(d_, item, query);
        list->insert(ip, i);
    }
}

// -----------------------------------------------------------------------------
void Linear::kmips_avg(             // k-mips (for avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query,                // query (user) vector
    float  &mmr,                        // mmr (return)
    float  &mean_k_ip,                  // mean of k_ip (return)
    float  &mean_pair_ip,               // mean pair-wise ip (return)
    Result *res)                        // result set S (return)
{
    // k-mips of query
    MaxK_List *list = new MaxK_List(k);
    linear_scan(query, list);
    
    // calc mmr, mean_k_ip, and mean_pair_ip
    mean_pair_ip = 0.0;             // init mean pair-wise inner product
    mean_k_ip = 0.0;                // sum of ip in S
    
    for (int i = 0; i < k; ++i) {
        int   tid = list->ith_id(i);
        float ip  = list->ith_key(i);
        mean_k_ip += ip; res[i].id_ = tid; res[i].key_ = ip;
        
        if (i == 0) continue;
        const float *target = item_set_ + (u64) tid*d_;
        for (int j = 0; j < i; ++j) {
            int did = list->ith_id(j);
            const float *dest = item_set_ + (u64) did*d_;
            float ip = calc_inner_product(d_, target, dest);
            
            mean_pair_ip += ip;
        }
    }
    mean_k_ip /= k;
    mean_pair_ip = mean_pair_ip*2/(k*(k-1));
    mmr = lambda*mean_k_ip - c*(1-lambda)*mean_pair_ip;
    
    delete list;                    // release space
}

// -----------------------------------------------------------------------------
void Linear::kmips_max(             // k-mips (for max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query,                // query (user) vector
    float  &mmr,                        // mmr (return)
    float  &mean_k_ip,                  // mean of k_ip (return)
    float  &max_pair_ip,                // max pair-wise inner product (return)
    Result *res)                        // result set S (return)
{
    // k-mips of query
    MaxK_List *list = new MaxK_List(k);
    linear_scan(query, list);
    
    // calc mmr, mean_k_ip, and max_pair_ip
    max_pair_ip = MINREAL;          // init maximum pair-wise inner product
    mean_k_ip = 0.0;                // sum of ip in S
    
    for (int i = 0; i < k; ++i) {
        int   tid = list->ith_id(i);
        float ip  = list->ith_key(i);
        mean_k_ip += ip; res[i].id_ = tid; res[i].key_ = ip;
        
        if (i == 0) continue;
        const float *target = item_set_ + (u64) tid*d_;
        for (int j = 0; j < i; ++j) {
            int did = list->ith_id(j);
            const float *dest = item_set_ + (u64) did*d_;
            float ip = calc_inner_product(d_, target, dest);
            
            if (ip > max_pair_ip) max_pair_ip = ip;
        }
    }
    mean_k_ip /= k;
    mmr = lambda*mean_k_ip - c*(1-lambda)*max_pair_ip;
    
    delete list;                    // release space
}

// -----------------------------------------------------------------------------
void Linear::kmips_avg_i2v(         // k-mips (for avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query,                // query (user) vector
    float  &mmr,                        // mmr (return)
    float  &mean_k_ip,                  // mean of k_ip (return)
    float  &mean_pair_ip,               // mean pair-wise ip (return)
    Result *res)                        // result set S (return)
{
    // k-mips of query
    MaxK_List *list = new MaxK_List(k);
    linear_scan(query, list);
    
    // calc mmr, mean_k_ip, and mean_pair_ip
    mean_pair_ip = 0.0;             // init mean pair-wise inner product
    mean_k_ip = 0.0;                // sum of ip in S
    
    for (int i = 0; i < k; ++i) {
        int   tid = list->ith_id(i);
        float ip  = list->ith_key(i);
        mean_k_ip += ip; res[i].id_ = tid; res[i].key_ = ip;
        
        if (i == 0) continue;
        const float *target = i2v_set_ + (u64) tid*d2_;
        for (int j = 0; j < i; ++j) {
            int   did = list->ith_id(j);
            const float *dest = i2v_set_ + (u64) did*d2_;
            float ip = calc_inner_product(d2_, target, dest);
            
            mean_pair_ip += ip;
        }
    }
    mean_k_ip /= k;
    mean_pair_ip = mean_pair_ip*2/(k*(k-1));
    mmr = lambda*mean_k_ip - c*(1-lambda)*mean_pair_ip;
    
    delete list;                    // release space
}

// -----------------------------------------------------------------------------
void Linear::kmips_max_i2v(         // k-mips (for max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query,                // query (user) vector
    float  &mmr,                        // mmr (return)
    float  &mean_k_ip,                  // mean of k_ip (return)
    float  &max_pair_ip,                // max pair-wise inner product (return)
    Result *res)                        // result set S (return)
{
    // k-mips of query
    MaxK_List *list = new MaxK_List(k);
    linear_scan(query, list);
    
    // calc mmr, mean_k_ip, and max_pair_ip
    max_pair_ip = MINREAL;          // init maximum pair-wise inner product
    mean_k_ip = 0.0;                // sum of ip in S
    
    for (int i = 0; i < k; ++i) {
        int   tid = list->ith_id(i);
        float ip  = list->ith_key(i);
        mean_k_ip += ip; res[i].id_ = tid; res[i].key_ = ip;
        
        if (i == 0) continue;
        const float *target = i2v_set_ + (u64) tid*d2_;
        for (int j = 0; j < i; ++j) {
            int   did = list->ith_id(j);
            const float *dest = i2v_set_ + (u64) did*d2_;
            float ip = calc_inner_product(d2_, target, dest);
            
            if (ip > max_pair_ip) max_pair_ip = ip;
        }
    }
    mean_k_ip /= k;
    mmr = lambda*mean_k_ip - c*(1-lambda)*max_pair_ip;
    
    delete list;                    // release space
}


} // end namespace ip
