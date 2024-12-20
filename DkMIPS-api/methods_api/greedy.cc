#include "greedy.h"

namespace ip {

// -----------------------------------------------------------------------------
Greedy::Greedy(                     // constructor
    int   n,                            // item cardinality
    int   d,                            // dimensionality
    int   d2,                           // dimensionality for 2nd space
    const float *item_set,              // item set by MF
    const float *i2v_set)               // i2v  set by item2vec
    : n_(n), d_(d), d2_(d2), item_set_(item_set), i2v_set_(i2v_set)
{
    mf_norm_  = new float[n];
    for (int i = 0; i < n; ++i) {
        const float *item = item_set + (u64) i*d;
        mf_norm_[i] = sqrt(calc_inner_product(d, item, item));
    }
    
    // init index_ and sort it in descending order of mf_norm_
    index_ = new int[n];
    int i = 0;
    std::iota(index_, index_+n, i++);
    std::sort(index_, index_+n, [&](int i,int j) { return mf_norm_[i] > 
        mf_norm_[j]; });
}

// -----------------------------------------------------------------------------
Greedy::~Greedy()                   // destructor
{
    if (index_   != nullptr) { delete[] index_;   index_   = nullptr; }
    if (mf_norm_ != nullptr) { delete[] mf_norm_; mf_norm_ = nullptr; }
}

// -----------------------------------------------------------------------------
int* Greedy::dkmips_avg(            // diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added,  false, sizeof(bool)*n_);
    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];

    float tau = MINREAL;            // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        if (ip[i] > tau) { tau = ip[i]; max_id = i; }
    }
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    mean_k_ip = tau; mean_pair_ip = 0.0;
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float factor = 2*c*(1-lambda)/(k-1);
    for (int j = 1; j < k; ++j) {
        // evaluate the score f(p,S) for each p \in P
        float local_sum = -1.0f; tau = MINREAL; max_id = -1;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the item that are added into S
            
            // calc the score f(p,S) and update the max score
            float sum_ip = calc_sum_ip(i, j, d_, res, item_set_);
            float score = lambda*ip[i] - factor*sum_ip;
            if (score > tau) { tau=score; max_id=i; local_sum=sum_ip; }
        }
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true;
        mean_k_ip += ip[max_id]; mean_pair_ip += local_sum;
    }
    mean_k_ip /= k;
    mean_pair_ip = mean_pair_ip*2/(k*(k-1));
    mmr = lambda*mean_k_ip - c*(1-lambda)*mean_pair_ip;
    
    int *result = new int[k];   
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    // release space
    delete[] added; delete[] ip; delete[] res;

    return result;
}

// -----------------------------------------------------------------------------
float Greedy::calc_sum_ip(          // calc sum of ip in results
    int    id,                          // item id
    int    n,                           // number of results
    int    d,                           // item dimension from MF/item2vec
    Result *res,                        // results
    const float *items)                 // items from MF/item2vec
{
    const float *query = items + (u64) id*d;
    
    float sum_ip = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float *item = items + (u64) res[i].id_*d;
        float ip = calc_inner_product(d, item, query);
        sum_ip += ip;
    }
    return sum_ip;
}

// -----------------------------------------------------------------------------
int* Greedy::dkmips_plus_avg(       // diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    float *sum_ip = new float[n_];  // sum of inner product of p \in P and S
    memset(added,  false, sizeof(bool)*n_);
    memset(sum_ip, 0.0f,  sizeof(float)*n_);
    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];
    
    float tau = MINREAL;            // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        if (ip[i] > tau) { tau = ip[i]; max_id = i; }
    }
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    mean_k_ip = tau; mean_pair_ip = 0.0;
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float factor = 2*c*(1-lambda)/(k-1);
    for (int j = 1; j < k; ++j) {
        // update sum_ip between each p \in P and S
        const float *added_item = item_set_ + (u64) max_id*d_;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the items that are added into S
            
            const float *item = item_set_ + (u64) i*d_;
            float ip_val = calc_inner_product(d_, item, added_item);
            sum_ip[i] += ip_val;
        }
        // evaluate the score f(p,S) for each p \in P
        float local_sum = -1.0f; tau = MINREAL; max_id = -1;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the item that are added into S
            
            // calc the score f(p, S) and update the max score
            float score = lambda*ip[i] - factor*sum_ip[i];
            if (score > tau) { tau=score; max_id=i; local_sum=sum_ip[i]; }
        }
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true;
        mean_k_ip += ip[max_id]; mean_pair_ip += local_sum;
    }
    mean_k_ip /= k;
    mean_pair_ip = mean_pair_ip*2/(k*(k-1));
    mmr = lambda*mean_k_ip - c*(1-lambda)*mean_pair_ip;
    
    int *result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    // release space
    delete[] added; delete[] ip; delete[] sum_ip; delete[] res;

    return result;
}

// -----------------------------------------------------------------------------
int* Greedy::dkmips_max(            // diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    float tau = MINREAL;            // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        if (ip[i] > tau) { tau = ip[i]; max_id = i; }
    }
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    mean_k_ip = tau; max_pair_ip = 0; // init 0 as there is no pair
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    for (int j = 1; j < k; ++j) {
        // evaluate the score f(p,S) for each p \in P
        float local_max = -1.0f; tau = MINREAL; max_id = -1;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the item that are added into S
            
            // calc the score f(p, S) and update the max score
            float max_ip = calc_max_ip(i, j, d_, res, item_set_);
            float this_max = std::max(max_pair_ip, max_ip);
            float score = f1*ip[i] - f2*(this_max-max_pair_ip);
            if (score > tau) { tau=score; max_id=i; local_max=this_max; }
        }
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true; 
        mean_k_ip += ip[max_id]; max_pair_ip = local_max;
    }
    mean_k_ip /= k;
    mmr = lambda*mean_k_ip - c*(1-lambda)*max_pair_ip;
    
    int *result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    // release space
    delete[] added; delete[] ip; delete[] res;

    return result;
}

// -----------------------------------------------------------------------------
float Greedy::calc_max_ip(          // calc max ip in results
    int    id,                          // item id
    int    n,                           // number of results
    int    d,                           // item dimension from MF/item2vec
    Result *res,                        // results
    const float *items)                 // items from MF/item2vec
{
    const float *query = items + (u64) id*d;
    
    float tau = MINREAL;
    for (int i = 0; i < n; ++i) {
        const float *item = items + (u64) res[i].id_*d;
        float ip = calc_inner_product(d, item, query);
        if (ip > tau) tau = ip;
    }
    return tau;
}

// -----------------------------------------------------------------------------
int* Greedy::dkmips_plus_max(       // diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    float *max_ip = new float[n_];  // max ip of each p \in P and S
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    float tau = MINREAL;            // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    for (int i = 0; i < n_; ++i) {
        max_ip[i] = MINREAL;
        
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        if (ip[i] > tau) { tau = ip[i]; max_id = i; }
    }
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    mean_k_ip = tau; max_pair_ip = 0; // init 0 as there is no pair
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    for (int j = 1; j < k; ++j) {
        // update max_ip between each p \in P and S
        const float *added_item = item_set_ + (u64) max_id*d_;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the items that are added into S
            
            const float *item = item_set_ + (u64) i*d_;
            float ip_val = calc_inner_product(d_, item, added_item);
            if (ip_val > max_ip[i]) max_ip[i] = ip_val;
        }
        // evaluate the score f(p,S) for each p \in P
        float local_max = -1.0f; tau = MINREAL; max_id = -1;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the item that are added into S
            
            // calc the score f(p, S) and update the max score
            float this_max = std::max(max_pair_ip, max_ip[i]);
            float score = f1*ip[i] - f2*(this_max-max_pair_ip);
            if (score > tau) { tau = score; max_id = i; local_max = this_max; }
        }
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true; 
        mean_k_ip += ip[max_id]; max_pair_ip = local_max;
    }
    mean_k_ip /= k;
    mmr = lambda*mean_k_ip - c*(1-lambda)*max_pair_ip;
    
    int *result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    // release space
    delete[] added; delete[] ip; delete[] max_ip; delete[] res;

    return result;
}




// -----------------------------------------------------------------------------
//  the following four methods consider two spaces
// -----------------------------------------------------------------------------
int* Greedy::dkmips_avg_i2v(        // diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];
    float tau = MINREAL;            // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        if (ip[i] > tau) { tau = ip[i]; max_id = i; }
    }
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    mean_k_ip = tau; mean_pair_ip = 0.0;
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float factor = 2*c*(1-lambda)/(k-1);
    for (int j = 1; j < k; ++j) {
        // evaluate the score f(p,S) for each p \in P
        float local_sum = -1.0f; tau = MINREAL; max_id = -1;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the item that are added into S
            
            // calc the score f(p, S) and update the max score
            float sum_ip = calc_sum_ip(i, j, d2_, res, i2v_set_);
            float score = lambda*ip[i] - factor*sum_ip;
            if (score > tau) { tau=score; max_id=i; local_sum=sum_ip; }
        }
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true;
        mean_k_ip += ip[max_id]; mean_pair_ip += local_sum;
    }
    mean_k_ip /= k;
    mean_pair_ip = mean_pair_ip*2/(k*(k-1));
    mmr = lambda*mean_k_ip - c*(1-lambda)*mean_pair_ip;
    
    int *result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    // release space
    delete[] added; delete[] ip; delete[] res;

    return result;
}

// -----------------------------------------------------------------------------
int* Greedy::dkmips_plus_avg_i2v(   // diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    float *sum_ip = new float[n_];  // sum of inner product of p \in P and S
    memset(added,  false, sizeof(bool)*n_);
    memset(sum_ip, 0.0f,  sizeof(float)*n_);
    
    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];
    float tau = MINREAL;            // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        if (ip[i] > tau) { tau = ip[i]; max_id = i; }
    }
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    mean_k_ip = tau; mean_pair_ip = 0.0;
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float factor = 2*c*(1-lambda)/(k-1);
    for (int j = 1; j < k; ++j) {
        // update sum_ip between each p \in P and S
        const float *added_item = i2v_set_ + (u64) max_id*d2_;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the items that are added into S
            
            const float *item = i2v_set_ + (u64) i*d2_;
            float ip_val = calc_inner_product(d2_, item, added_item);
            sum_ip[i] += ip_val;
        }
        // evaluate the score f(p,S) for each p \in P
        float local_sum = -1.0f; tau = MINREAL; max_id = -1;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the item that are added into S
            
            // calc the score f(p, S) and update the max score
            float score = lambda*ip[i] - factor*sum_ip[i];
            if (score > tau) { tau=score; max_id=i; local_sum=sum_ip[i]; }
        }
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true;
        mean_k_ip += ip[max_id]; mean_pair_ip += local_sum;
    }
    mean_k_ip /= k;
    mean_pair_ip = mean_pair_ip*2/(k*(k-1));
    mmr = lambda*mean_k_ip - c*(1-lambda)*mean_pair_ip;
    
    int *result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    // release space
    delete[] added; delete[] ip; delete[] sum_ip; delete[] res;

    return result;
}

// -----------------------------------------------------------------------------
int* Greedy::dkmips_max_i2v(        // diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    float tau = MINREAL;            // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        if (ip[i] > tau) { tau = ip[i]; max_id = i; }
    }
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    mean_k_ip = tau; max_pair_ip = 0; // init 0 as there is no pair
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    for (int j = 1; j < k; ++j) {
        // evaluate the score f(p,S) for each p \in P
        float local_max = -1.0f; tau = MINREAL; max_id = -1;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the item that are added into S
            
            // calc the score f(p, S) and update the max score
            float max_ip = calc_max_ip(i, j, d2_, res, i2v_set_);
            float this_max = std::max(max_pair_ip, max_ip);
            float score = f1*ip[i] - f2*(this_max-max_pair_ip);
            if (score > tau) { tau=score; max_id=i; local_max=this_max; }
        }
        // if (tau < 0) { printf("j=%d\n", j); break; }
        
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true; 
        mean_k_ip += ip[max_id]; max_pair_ip = local_max;
    }
    mean_k_ip /= k;
    mmr = lambda*mean_k_ip - c*(1-lambda)*max_pair_ip;
    
    int *result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    // release space
    delete[] added; delete[] ip; delete[] res; 

    return result;
}

// -----------------------------------------------------------------------------
int* Greedy::dkmips_plus_max_i2v(   // diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    float *max_ip = new float[n_];  // max ip of each p \in P and S
    memset(added, false, sizeof(bool)*n_);
    
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    float tau = MINREAL;            // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    for (int i = 0; i < n_; ++i) {
        max_ip[i] = MINREAL;
        
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        if (ip[i] > tau) { tau = ip[i]; max_id = i; }
    }
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    mean_k_ip = tau; max_pair_ip = 0; // init 0 as there is no pair
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    for (int j = 1; j < k; ++j) {
        // update max_ip between each p \in P and S
        const float *added_item = i2v_set_ + (u64) max_id*d2_;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the items that are added into S
            
            const float *item = i2v_set_ + (u64) i*d2_;
            float ip_val = calc_inner_product(d2_, item, added_item);
            if (ip_val > max_ip[i]) max_ip[i] = ip_val;
        }
        // evaluate the score f(p,S) for each p \in P
        float local_max = -1.0f; tau = MINREAL; max_id = -1;
        for (int i = 0; i < n_; ++i) {
            if (added[i]) continue; // skip the item that are added into S
            
            // calc the score f(p, S) and update the max score
            float this_max = std::max(max_pair_ip, max_ip[i]);
            float score = f1*ip[i] - f2*(this_max-max_pair_ip);
            if (score > tau) { tau = score; max_id = i; local_max = this_max; }
        }
        // if (tau < 0) { printf("j=%d\n", j); break; }
        
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true; 
        mean_k_ip += ip[max_id]; max_pair_ip = local_max;
    }
    mean_k_ip /= k;
    mmr = lambda*mean_k_ip - c*(1-lambda)*max_pair_ip;
    
    int *result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    // release space
    delete[] added; delete[] ip; delete[] max_ip; delete[] res;

    return result;
}

} // end namespace ip
