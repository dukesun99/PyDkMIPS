#include "dual_greedy.h"

namespace ip {

// -----------------------------------------------------------------------------
Dual_Greedy::Dual_Greedy(           // constructor
    int   n,                            // item cardinality
    int   d,                            // dimensionality
    int   d2,                           // dimensionality for 2nd space
    const float *item_set,              // item set by MF
    const float *i2v_set)               // i2v  set by item2vec
    : n_(n), d_(d), d2_(d2), item_set_(item_set), i2v_set_(i2v_set)
{
    mf_norm_ = new float[n];
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
Dual_Greedy::~Dual_Greedy()         // destructor
{
    if (index_   != nullptr) { delete[] index_;   index_   = nullptr; }
    if (mf_norm_ != nullptr) { delete[] mf_norm_; mf_norm_ = nullptr; }
}

// -----------------------------------------------------------------------------
int* Dual_Greedy::dkmips_avg(       // diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];

    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1 = new Result[k];
    Result *res_2 = new Result[k];
    
    MaxK_List *list = new MaxK_List(2);
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        list->insert(ip[i], i);
    }
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
    
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    float mean_k_ip_1 = tau_1, mean_pair_ip_1 = 0.0f;
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    float mean_k_ip_2 = tau_2, mean_pair_ip_2 = 0.0f;
    
    // -------------------------------------------------------------------------
    //  step 2: add j \in [2,k] item p^j into S1 and S2
    // -------------------------------------------------------------------------
    float factor = 2*c*(1-lambda)/(k-1);
    float local_sum_1 = 0.0f, local_sum_2 = 0.0f;
    int   cnt_1 = 1, cnt_2 = 1;
    
    while (cnt_1 < k || cnt_2 < k) {
        // evaluate the score f(p, S1) for each p \in P\(S1 and S2)
        local_sum_1 = -1.0f; tau_1 = MINREAL; max_id_1 = -1;
        if (cnt_1 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S1) and update tau_1
                float sum_ip = calc_sum_ip(i, cnt_1, d_, res_1, item_set_);
                float score = lambda*ip[i] - factor*sum_ip;
                if (score > tau_1) { tau_1=score; max_id_1=i; local_sum_1=sum_ip; }
            }
        }
        // evaluate the score f(p, S2) for each p \in P\(S1 and S2)
        local_sum_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1;
        if (cnt_2 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2) 
                
                // calc the score f(p, S2) and update tau_2
                float sum_ip = calc_sum_ip(i, cnt_2, d_, res_2, item_set_);
                float score = lambda*ip[i] - factor*sum_ip;
                if (score > tau_2) { tau_2=score; max_id_2=i; local_sum_2=sum_ip; }
            }
        }
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1]; 
            mean_k_ip_1 += ip[max_id_1]; mean_pair_ip_1 += local_sum_1;
            added[max_id_1] = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2]; 
            mean_k_ip_2 += ip[max_id_2]; mean_pair_ip_2 += local_sum_2;
            added[max_id_2] = true; ++cnt_2;
        }
    }
    // -------------------------------------------------------------------------
    //  step 3: choose the larger f(S) and update S
    // -------------------------------------------------------------------------
    // calc f(S1)
    mean_k_ip_1 /= k; 
    mean_pair_ip_1 = mean_pair_ip_1*2/(k*(k-1));
    float mmr_1 = lambda*mean_k_ip_1 - c*(1-lambda)*mean_pair_ip_1;
    
    // calc f(S2)
    mean_k_ip_2 /= k;
    mean_pair_ip_2 = mean_pair_ip_2*2/(k*(k-1));
    float mmr_2 = lambda*mean_k_ip_2 - c*(1-lambda)*mean_pair_ip_2;
    
    // choose the larger f(S) and update S
    if (mmr_1 >= mmr_2) {
        mean_k_ip = mean_k_ip_1; mean_pair_ip = mean_pair_ip_1; mmr = mmr_1;
        for (int i = 0; i < k; ++i) res[i] = res_1[i];
    } else {
        mean_k_ip = mean_k_ip_2; mean_pair_ip = mean_pair_ip_2; mmr = mmr_2;
        for (int i = 0; i < k; ++i) res[i] = res_2[i];
    }
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }

    // release space
    delete[] added; delete[] ip; delete[] res_1; delete[] res_2; delete list; delete[] res;
    return result;
}

// -----------------------------------------------------------------------------
float Dual_Greedy::calc_sum_ip(     // calc sum of ip in results
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
int* Dual_Greedy::dkmips_plus_avg(  // diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];
    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1 = new Result[k];
    Result *res_2 = new Result[k];
    float *sum_ip_1 = new float[n_];  // sum of ip of p \in P and S1
    float *sum_ip_2 = new float[n_];  // sum of ip of p \in P and S2
    
    memset(sum_ip_1, 0.0f, sizeof(float)*n_);
    memset(sum_ip_2, 0.0f, sizeof(float)*n_);
    
    MaxK_List *list = new MaxK_List(2);
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        list->insert(ip[i], i);
    }
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
    
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    float mean_k_ip_1 = tau_1, mean_pair_ip_1 = 0.0f;
    bool  choose_1 = true;
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    float mean_k_ip_2 = tau_2, mean_pair_ip_2 = 0.0f;
    bool  choose_2 = true;
    
    // -------------------------------------------------------------------------
    //  step 2: add j \in [2,k] item p^j into S1 and S2
    // -------------------------------------------------------------------------
    float factor = 2*c*(1-lambda)/(k-1);
    float local_sum_1 = 0.0f, local_sum_2 = 0.0f;
    int   cnt_1 = 1, cnt_2 = 1;
    
    while (cnt_1 < k || cnt_2 < k) {
        // update sum_ip_1 between each p \in P\(S1 and S2) and S1
        if (choose_1) {
            const float *added_item = item_set_ + (u64) max_id_1*d_;
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                const float *item = item_set_ + (u64) i*d_;
                float ip_val = calc_inner_product(d_, item, added_item);
                sum_ip_1[i] += ip_val;
            }
        }
        // update sum_ip_2 between each p \in P\(S1 and S2) and S2
        if (choose_2) {
            const float *added_item = item_set_ + (u64) max_id_2*d_;
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                const float *item = item_set_ + (u64) i*d_;
                float ip_val = calc_inner_product(d_, item, added_item);
                sum_ip_2[i] += ip_val;
            }
        }
        
        // evaluate the score f(p, S1) for each p \in P\(S1 and S2)
        local_sum_1 = -1.0f; tau_1 = MINREAL; max_id_1 = -1; choose_1 = false;
        if (cnt_1 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S1) and update tau_1
                float score = lambda*ip[i] - factor*sum_ip_1[i];
                if (score > tau_1) { tau_1=score; max_id_1=i; local_sum_1=sum_ip_1[i]; }
            }
        }
        // evaluate the score f(p, S2) for each p \in P\(S1 and S2)
        local_sum_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1; choose_2 = false;
        if (cnt_2 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2) 
                
                // calc the score f(p, S2) and update tau_2
                float score = lambda*ip[i] - factor*sum_ip_2[i];
                if (score > tau_2) { tau_2=score; max_id_2=i; local_sum_2=sum_ip_2[i]; }
            }
        }
        
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1]; 
            mean_k_ip_1 += ip[max_id_1]; mean_pair_ip_1 += local_sum_1;
            added[max_id_1] = true; choose_1 = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2]; 
            mean_k_ip_2 += ip[max_id_2]; mean_pair_ip_2 += local_sum_2;
            added[max_id_2] = true; choose_2 = true; ++cnt_2;
        }
    }
    // -------------------------------------------------------------------------
    //  step 3: choose the larger f(S) and update S
    // -------------------------------------------------------------------------
    // calc f(S1)
    mean_k_ip_1 /= k; 
    mean_pair_ip_1 = mean_pair_ip_1*2/(k*(k-1));
    float mmr_1 = lambda*mean_k_ip_1 - c*(1-lambda)*mean_pair_ip_1;
    
    // calc f(S2)
    mean_k_ip_2 /= k;
    mean_pair_ip_2 = mean_pair_ip_2*2/(k*(k-1));
    float mmr_2 = lambda*mean_k_ip_2 - c*(1-lambda)*mean_pair_ip_2;
    
    // choose the larger f(S) and update S
    if (mmr_1 >= mmr_2) {
        mean_k_ip = mean_k_ip_1; mean_pair_ip = mean_pair_ip_1; mmr = mmr_1;
        for (int i = 0; i < k; ++i) res[i] = res_1[i];
    } else {
        mean_k_ip = mean_k_ip_2; mean_pair_ip = mean_pair_ip_2; mmr = mmr_2;
        for (int i = 0; i < k; ++i) res[i] = res_2[i];
    }
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    
    // release space
    delete[] added; delete[] ip; delete list;
    delete[] res_1; delete[] res_2; delete[] sum_ip_1; delete[] sum_ip_2; 
    delete[] res;
    return result;
}

// -----------------------------------------------------------------------------
int* Dual_Greedy::dkmips_max(       // diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1 = new Result[k];
    Result *res_2 = new Result[k];
    
    MaxK_List *list = new MaxK_List(2);
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        list->insert(ip[i], i);
    }
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
    
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    float mean_k_ip_1 = tau_1, max_pair_ip_1 = 0.0f;
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    float mean_k_ip_2 = tau_2, max_pair_ip_2 = 0.0f;
    
    // -------------------------------------------------------------------------
    //  step 2: add j \in [2,k] item p^j into S1 and S2
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    float local_max_1 = -1.0f, local_max_2 = -1.0f;
    int   cnt_1 = 1, cnt_2 = 1;
    
    while (cnt_1 < k || cnt_2 < k) {
        // evaluate the score f(p,S1) for each p \in P\(S1 and S2)
        local_max_1 = -1.0f; tau_1 = MINREAL; max_id_1 = -1;
        if (cnt_1 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S1) and update tau_1
                float max_ip = calc_max_ip(i, cnt_1, d_, res_1, item_set_);
                float this_max = std::max(max_pair_ip_1, max_ip);
                float score = f1*ip[i] - f2*(this_max-max_pair_ip_1);
                if (score > tau_1) { tau_1=score; max_id_1=i; local_max_1=this_max; }
            }
        }
        // evaluate the score f(p,S2) for each p \in P\(S1 and S2)
        local_max_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1;
        if (cnt_2 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S2) and update tau_2
                float max_ip = calc_max_ip(i, cnt_2, d_, res_2, item_set_);
                float this_max = std::max(max_pair_ip_2, max_ip);
                float score = f1*ip[i] - f2*(this_max-max_pair_ip_2);
                if (score > tau_2) { tau_2=score; max_id_2=i; local_max_2=this_max; }
            }
        }
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1]; 
            mean_k_ip_1 += ip[max_id_1]; max_pair_ip_1 = local_max_1;
            added[max_id_1] = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2]; 
            mean_k_ip_2 += ip[max_id_2]; max_pair_ip_2 = local_max_2;
            added[max_id_2] = true; ++cnt_2;
        }
    }
    // -------------------------------------------------------------------------
    //  step 3: choose the larger f(S) and update S
    // -------------------------------------------------------------------------
    // calc f(S1)
    mean_k_ip_1 /= k;
    float mmr_1 = lambda*mean_k_ip_1 - c*(1-lambda)*max_pair_ip_1;
    
    // calc f(S2)
    mean_k_ip_2 /= k;
    float mmr_2 = lambda*mean_k_ip_2 - c*(1-lambda)*max_pair_ip_2;
    
    // choose the larger f(S) and update S
    if (mmr_1 >= mmr_2) {
        mean_k_ip = mean_k_ip_1; max_pair_ip = max_pair_ip_1; mmr = mmr_1;
        for (int i = 0; i < k; ++i) res[i] = res_1[i];
    } else {
        mean_k_ip = mean_k_ip_2; max_pair_ip = max_pair_ip_2; mmr = mmr_2;
        for (int i = 0; i < k; ++i) res[i] = res_2[i];
    }
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    
    // release space
    delete[] added; delete[] ip; delete[] res_1; delete[] res_2; delete list; delete[] res;
    return result;
}

// -----------------------------------------------------------------------------
float Dual_Greedy::calc_max_ip(     // calc max ip in results
    int    id,                          // item id
    int    n,                           // number of results
    int    d,                           // item dimension from MF/item2vec
    Result *res,                        // results
    const float *items)                 // items from MF/item2vec
{
    const float *query = items + (u64) id*d;
    
    float max_ip = MINREAL;
    for (int i = 0; i < n; ++i) {
        const float *item = items + (u64) res[i].id_*d;
        float ip = calc_inner_product(d, item, query);
        if (ip > max_ip) max_ip = ip;
    }
    return max_ip;
}

// -----------------------------------------------------------------------------
int* Dual_Greedy::dkmips_plus_max(  // diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1 = new Result[k];
    Result *res_2 = new Result[k];
    float *max_ip_1 = new float[n_];  // max ip of p \in P and S
    float *max_ip_2 = new float[n_];  // max ip of p \in P and S
    
    for (int i = 0; i < n_; ++i) { max_ip_1[i]=MINREAL; max_ip_2[i]=MINREAL; }
    
    MaxK_List *list = new MaxK_List(2);
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        list->insert(ip[i], i);
    }
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
    
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    float mean_k_ip_1 = tau_1, max_pair_ip_1 = 0.0f;
    bool  choose_1 = true;
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    float mean_k_ip_2 = tau_2, max_pair_ip_2 = 0.0f;
    bool  choose_2 = true;
    
    // -------------------------------------------------------------------------
    //  step 2: add j \in [2,k] item p^j into S1 and S2
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    float local_max_1 = -1.0f, local_max_2 = -1.0f;
    int   cnt_1 = 1, cnt_2 = 1;
    
    while (cnt_1 < k || cnt_2 < k) {
        // update max_ip_1 between each p \in P\(S1 and S2) and S1
        if (choose_1) {
            const float *added_item = item_set_ + (u64) max_id_1*d_;
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                const float *item = item_set_ + (u64) i*d_;
                float ip_val = calc_inner_product(d_, item, added_item);
                if (ip_val > max_ip_1[i]) max_ip_1[i] = ip_val;
            }
        }
        // update max_ip_2 between each p \in P\(S1 and S2) and S2
        if (choose_2) {
            const float *added_item = item_set_ + (u64) max_id_2*d_;
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                const float *item = item_set_ + (u64) i*d_;
                float ip_val = calc_inner_product(d_, item, added_item);
                if (ip_val > max_ip_2[i]) max_ip_2[i] = ip_val;
            }
        }
        
        // evaluate the score f(p,S1) for each p \in P\(S1 and S2)
        local_max_1 = -1.0f; tau_1 = MINREAL; max_id_1 = -1; choose_1 = false;
        if (cnt_1 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S1) and update tau_1
                float this_max = std::max(max_pair_ip_1, max_ip_1[i]);
                float score = f1*ip[i] - f2*(this_max-max_pair_ip_1);
                if (score > tau_1) { tau_1=score; max_id_1=i; local_max_1=this_max; }
            }
        }
        // evaluate the score f(p,S2) for each p \in P\(S1 and S2)
        local_max_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1; choose_2 = false;
        if (cnt_2 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S2) and update tau_2
                float this_max = std::max(max_pair_ip_2, max_ip_2[i]);
                float score = f1*ip[i] - f2*(this_max-max_pair_ip_2);
                if (score > tau_2) { tau_2=score; max_id_2=i; local_max_2=this_max; }
            }
        }
        
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1]; 
            mean_k_ip_1 += ip[max_id_1]; max_pair_ip_1 = local_max_1;
            added[max_id_1] = true; choose_1 = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2]; 
            mean_k_ip_2 += ip[max_id_2]; max_pair_ip_2 = local_max_2;
            added[max_id_2] = true; choose_2 = true; ++cnt_2;
        }
    }
    // -------------------------------------------------------------------------
    //  step 3: choose the larger f(S) and update S
    // -------------------------------------------------------------------------
    // calc f(S1)
    mean_k_ip_1 /= k;
    float mmr_1 = lambda*mean_k_ip_1 - c*(1-lambda)*max_pair_ip_1;
    
    // calc f(S2)
    mean_k_ip_2 /= k;
    float mmr_2 = lambda*mean_k_ip_2 - c*(1-lambda)*max_pair_ip_2;
    
    // choose the larger f(S) and update S
    if (mmr_1 >= mmr_2) {
        mean_k_ip = mean_k_ip_1; max_pair_ip = max_pair_ip_1; mmr = mmr_1;
        for (int i = 0; i < k; ++i) res[i] = res_1[i];
    } else {
        mean_k_ip = mean_k_ip_2; max_pair_ip = max_pair_ip_2; mmr = mmr_2;
        for (int i = 0; i < k; ++i) res[i] = res_2[i];
    }
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    
    // release space
    delete[] added; delete[] ip; delete list;
    delete[] res_1; delete[] res_2; delete[] max_ip_1; delete[] max_ip_2;
    delete[] res;
    return result;
}


// -----------------------------------------------------------------------------
//  the following four methods consider two spaces
// -----------------------------------------------------------------------------
int* Dual_Greedy::dkmips_avg_i2v(   // diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];
    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1 = new Result[k];
    Result *res_2 = new Result[k];
    
    MaxK_List *list = new MaxK_List(2);
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        list->insert(ip[i], i);
    }
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
    
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    float mean_k_ip_1 = tau_1, mean_pair_ip_1 = 0.0f;
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    float mean_k_ip_2 = tau_2, mean_pair_ip_2 = 0.0f;
    
    // -------------------------------------------------------------------------
    //  step 2: add j \in [2,k] item p^j into S1 and S2
    // -------------------------------------------------------------------------
    float factor = 2*c*(1-lambda)/(k-1);
    float local_sum_1 = 0.0f, local_sum_2 = 0.0f;
    int   cnt_1 = 1, cnt_2 = 1;
    
    while (cnt_1 < k || cnt_2 < k) {
        // evaluate the score f(p, S1) for each p \in P\(S1 and S2)
        local_sum_1 = -1.0f; tau_1 = MINREAL; max_id_1 = -1;
        if (cnt_1 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S1) and update tau_1
                float sum_ip = calc_sum_ip(i, cnt_1, d2_, res_1, i2v_set_);
                float score = lambda*ip[i] - factor*sum_ip;
                if (score > tau_1) { tau_1=score; max_id_1=i; local_sum_1=sum_ip; }
            }
        }
        // evaluate the score f(p, S2) for each p \in P\(S1 and S2)
        local_sum_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1;
        if (cnt_2 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2) 
                
                // calc the score f(p, S2) and update tau_2
                float sum_ip = calc_sum_ip(i, cnt_2, d2_, res_2, i2v_set_);
                float score = lambda*ip[i] - factor*sum_ip;
                if (score > tau_2) { tau_2=score; max_id_2=i; local_sum_2=sum_ip; }
            }
        }
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1]; 
            mean_k_ip_1 += ip[max_id_1]; mean_pair_ip_1 += local_sum_1;
            added[max_id_1] = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2]; 
            mean_k_ip_2 += ip[max_id_2]; mean_pair_ip_2 += local_sum_2;
            added[max_id_2] = true; ++cnt_2;
        }
    }
    // -------------------------------------------------------------------------
    //  step 3: choose the larger f(S) and update S
    // -------------------------------------------------------------------------
    // calc f(S1)
    mean_k_ip_1 /= k; 
    mean_pair_ip_1 = mean_pair_ip_1*2/(k*(k-1));
    float mmr_1 = lambda*mean_k_ip_1 - c*(1-lambda)*mean_pair_ip_1;
    
    // calc f(S2)
    mean_k_ip_2 /= k;
    mean_pair_ip_2 = mean_pair_ip_2*2/(k*(k-1));
    float mmr_2 = lambda*mean_k_ip_2 - c*(1-lambda)*mean_pair_ip_2;
    
    // choose the larger f(S) and update S
    if (mmr_1 >= mmr_2) {
        mean_k_ip = mean_k_ip_1; mean_pair_ip = mean_pair_ip_1; mmr = mmr_1;
        for (int i = 0; i < k; ++i) res[i] = res_1[i];
    } else {
        mean_k_ip = mean_k_ip_2; mean_pair_ip = mean_pair_ip_2; mmr = mmr_2;
        for (int i = 0; i < k; ++i) res[i] = res_2[i];
    }
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    
    // release space
    delete[] added; delete[] ip; delete[] res_1; delete[] res_2; delete list; delete[] res;
    return result;
}

// -----------------------------------------------------------------------------
int* Dual_Greedy::dkmips_plus_avg_i2v(// diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];
    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1 = new Result[k];
    Result *res_2 = new Result[k];
    float *sum_ip_1 = new float[n_];  // sum of ip of p \in P and S1
    float *sum_ip_2 = new float[n_];  // sum of ip of p \in P and S2
    
    memset(sum_ip_1, 0.0f, sizeof(float)*n_);
    memset(sum_ip_2, 0.0f, sizeof(float)*n_);
    
    MaxK_List *list = new MaxK_List(2);
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        list->insert(ip[i], i);
    }
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
    
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    float mean_k_ip_1 = tau_1, mean_pair_ip_1 = 0.0f;
    bool  choose_1 = true;
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    float mean_k_ip_2 = tau_2, mean_pair_ip_2 = 0.0f;
    bool  choose_2 = true;
    
    // -------------------------------------------------------------------------
    //  step 2: add j \in [2,k] item p^j into S1 and S2
    // -------------------------------------------------------------------------
    float factor = 2*c*(1-lambda)/(k-1);
    float local_sum_1 = 0.0f, local_sum_2 = 0.0f;
    int   cnt_1 = 1, cnt_2 = 1;
    
    while (cnt_1 < k || cnt_2 < k) {
        // update sum_ip_1 between each p \in P\(S1 and S2) and S1
        if (choose_1) {
            const float *added_item = i2v_set_ + (u64) max_id_1*d2_;
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                const float *item = i2v_set_ + (u64) i*d2_;
                float ip_val = calc_inner_product(d2_, item, added_item);
                sum_ip_1[i] += ip_val;
            }
        }
        // update sum_ip_2 between each p \in P\(S1 and S2) and S2
        if (choose_2) {
            const float *added_item = i2v_set_ + (u64) max_id_2*d2_;
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                const float *item = i2v_set_ + (u64) i*d2_;
                float ip_val = calc_inner_product(d2_, item, added_item);
                sum_ip_2[i] += ip_val;
            }
        }
        
        // evaluate the score f(p, S1) for each p \in P\(S1 and S2)
        local_sum_1 = -1.0f; tau_1 = MINREAL; max_id_1 = -1; choose_1 = false;
        if (cnt_1 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S1) and update tau_1
                float score = lambda*ip[i] - factor*sum_ip_1[i];
                if (score > tau_1) { tau_1=score; max_id_1=i; local_sum_1=sum_ip_1[i]; }
            }
        }
        // evaluate the score f(p, S2) for each p \in P\(S1 and S2)
        local_sum_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1; choose_2 = false;
        if (cnt_2 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2) 
                
                // calc the score f(p, S2) and update tau_2
                float score = lambda*ip[i] - factor*sum_ip_2[i];
                if (score > tau_2) { tau_2=score; max_id_2=i; local_sum_2=sum_ip_2[i]; }
            }
        }
        
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1]; 
            mean_k_ip_1 += ip[max_id_1]; mean_pair_ip_1 += local_sum_1;
            added[max_id_1] = true; choose_1 = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2]; 
            mean_k_ip_2 += ip[max_id_2]; mean_pair_ip_2 += local_sum_2;
            added[max_id_2] = true; choose_2 = true; ++cnt_2;
        }
    }
    // -------------------------------------------------------------------------
    //  step 3: choose the larger f(S) and update S
    // -------------------------------------------------------------------------
    // calc f(S1)
    mean_k_ip_1 /= k; 
    mean_pair_ip_1 = mean_pair_ip_1*2/(k*(k-1));
    float mmr_1 = lambda*mean_k_ip_1 - c*(1-lambda)*mean_pair_ip_1;
    
    // calc f(S2)
    mean_k_ip_2 /= k;
    mean_pair_ip_2 = mean_pair_ip_2*2/(k*(k-1));
    float mmr_2 = lambda*mean_k_ip_2 - c*(1-lambda)*mean_pair_ip_2;
    
    // choose the larger f(S) and update S
    if (mmr_1 >= mmr_2) {
        mean_k_ip = mean_k_ip_1; mean_pair_ip = mean_pair_ip_1; mmr = mmr_1;
        for (int i = 0; i < k; ++i) res[i] = res_1[i];
    } else {
        mean_k_ip = mean_k_ip_2; mean_pair_ip = mean_pair_ip_2; mmr = mmr_2;
        for (int i = 0; i < k; ++i) res[i] = res_2[i];
    }
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    
    // release space
    delete[] added; delete[] ip; delete list;
    delete[] res_1; delete[] res_2; delete[] sum_ip_1; delete[] sum_ip_2; 
    delete[] res;
    return result;
}

// -----------------------------------------------------------------------------
int* Dual_Greedy::dkmips_max_i2v(   // diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1 = new Result[k];
    Result *res_2 = new Result[k];
    
    MaxK_List *list = new MaxK_List(2);
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        list->insert(ip[i], i);
    }
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
    
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    float mean_k_ip_1 = tau_1, max_pair_ip_1 = 0.0f;
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    float mean_k_ip_2 = tau_2, max_pair_ip_2 = 0.0f;
    
    // -------------------------------------------------------------------------
    //  step 2: add j \in [2,k] item p^j into S1 and S2
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    float local_max_1 = -1.0f, local_max_2 = -1.0f;
    int   cnt_1 = 1, cnt_2 = 1;
    
    while (cnt_1 < k || cnt_2 < k) {
        // evaluate the score f(p,S1) for each p \in P\(S1 and S2)
        local_max_1 = -1.0f; tau_1 = MINREAL; max_id_1 = -1;
        if (cnt_1 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S1) and update tau_1
                float max_ip = calc_max_ip(i, cnt_1, d2_, res_1, i2v_set_);
                float this_max = std::max(max_pair_ip_1, max_ip);
                float score = f1*ip[i] - f2*(this_max-max_pair_ip_1);
                if (score > tau_1) { tau_1=score; max_id_1=i; local_max_1=this_max; }
            }
        }
        // evaluate the score f(p,S2) for each p \in P\(S1 and S2)
        local_max_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1;
        if (cnt_2 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S2) and update tau_2
                float max_ip = calc_max_ip(i, cnt_2, d2_, res_2, i2v_set_);
                float this_max = std::max(max_pair_ip_2, max_ip);
                float score = f1*ip[i] - f2*(this_max-max_pair_ip_2);
                if (score > tau_2) { tau_2=score; max_id_2=i; local_max_2=this_max; }
            }
        }
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1]; 
            mean_k_ip_1 += ip[max_id_1]; max_pair_ip_1 = local_max_1;
            added[max_id_1] = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2]; 
            mean_k_ip_2 += ip[max_id_2]; max_pair_ip_2 = local_max_2;
            added[max_id_2] = true; ++cnt_2;
        }
    }
    // -------------------------------------------------------------------------
    //  step 3: choose the larger f(S) and update S
    // -------------------------------------------------------------------------
    // calc f(S1)
    mean_k_ip_1 /= k;
    float mmr_1 = lambda*mean_k_ip_1 - c*(1-lambda)*max_pair_ip_1;
    
    // calc f(S2)
    mean_k_ip_2 /= k;
    float mmr_2 = lambda*mean_k_ip_2 - c*(1-lambda)*max_pair_ip_2;
    
    // choose the larger f(S) and update S
    if (mmr_1 >= mmr_2) {
        mean_k_ip = mean_k_ip_1; max_pair_ip = max_pair_ip_1; mmr = mmr_1;
        for (int i = 0; i < k; ++i) res[i] = res_1[i];
    } else {
        mean_k_ip = mean_k_ip_2; max_pair_ip = max_pair_ip_2; mmr = mmr_2;
        for (int i = 0; i < k; ++i) res[i] = res_2[i];
    }
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    
    // release space
    delete[] added; delete[] ip; delete[] res_1; delete[] res_2; delete list; delete[] res;
    return result;
}

// -----------------------------------------------------------------------------
int* Dual_Greedy::dkmips_plus_max_i2v(// diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *added  = new bool[n_];   // whether each p \in P is added to S
    float *ip     = new float[n_];  // ip between each p \in P and q
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1 = new Result[k];
    Result *res_2 = new Result[k];
    float *max_ip_1 = new float[n_];  // max ip of p \in P and S
    float *max_ip_2 = new float[n_];  // max ip of p \in P and S
    
    for (int i = 0; i < n_; ++i) { max_ip_1[i]=MINREAL; max_ip_2[i]=MINREAL; }
    
    MaxK_List *list = new MaxK_List(2);
    for (int i = 0; i < n_; ++i) {
        const float *item = item_set_ + (u64) i*d_;
        ip[i] = calc_inner_product(d_, item, query);
        list->insert(ip[i], i);
    }
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
    
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    float mean_k_ip_1 = tau_1, max_pair_ip_1 = 0.0f;
    bool  choose_1 = true;
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    float mean_k_ip_2 = tau_2, max_pair_ip_2 = 0.0f;
    bool  choose_2 = true;
    
    // -------------------------------------------------------------------------
    //  step 2: add j \in [2,k] item p^j into S1 and S2
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    float local_max_1 = -1.0f, local_max_2 = -1.0f;
    int   cnt_1 = 1, cnt_2 = 1;
    
    while (cnt_1 < k || cnt_2 < k) {
        // update max_ip_1 between each p \in P\(S1 and S2) and S1
        if (choose_1) {
            const float *added_item = i2v_set_ + (u64) max_id_1*d2_;
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                const float *item = i2v_set_ + (u64) i*d2_;
                float ip_val = calc_inner_product(d2_, item, added_item);
                if (ip_val > max_ip_1[i]) max_ip_1[i] = ip_val;
            }
        }
        // update max_ip_2 between each p \in P\(S1 and S2) and S2
        if (choose_2) {
            const float *added_item = i2v_set_ + (u64) max_id_2*d2_;
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                const float *item = i2v_set_ + (u64) i*d2_;
                float ip_val = calc_inner_product(d2_, item, added_item);
                if (ip_val > max_ip_2[i]) max_ip_2[i] = ip_val;
            }
        }
        
        // evaluate the score f(p,S1) for each p \in P\(S1 and S2)
        local_max_1 = -1.0f; tau_1 = MINREAL; max_id_1 = -1; choose_1 = false;
        if (cnt_1 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S1) and update tau_1
                float this_max = std::max(max_pair_ip_1, max_ip_1[i]);
                float score = f1*ip[i] - f2*(this_max-max_pair_ip_1);
                if (score > tau_1) { tau_1=score; max_id_1=i; local_max_1=this_max; }
            }
        }
        // evaluate the score f(p,S2) for each p \in P\(S1 and S2)
        local_max_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1; choose_2 = false;
        if (cnt_2 < k) {
            for (int i = 0; i < n_; ++i) {
                if (added[i]) continue; // skip items added into (S1 and S2)
                
                // calc the score f(p, S2) and update tau_2
                float this_max = std::max(max_pair_ip_2, max_ip_2[i]);
                float score = f1*ip[i] - f2*(this_max-max_pair_ip_2);
                if (score > tau_2) { tau_2=score; max_id_2=i; local_max_2=this_max; }
            }
        }
        
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1]; 
            mean_k_ip_1 += ip[max_id_1]; max_pair_ip_1 = local_max_1;
            added[max_id_1] = true; choose_1 = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2]; 
            mean_k_ip_2 += ip[max_id_2]; max_pair_ip_2 = local_max_2;
            added[max_id_2] = true; choose_2 = true; ++cnt_2;
        }
    }
    // -------------------------------------------------------------------------
    //  step 3: choose the larger f(S) and update S
    // -------------------------------------------------------------------------
    // calc f(S1)
    mean_k_ip_1 /= k;
    float mmr_1 = lambda*mean_k_ip_1 - c*(1-lambda)*max_pair_ip_1;
    
    // calc f(S2)
    mean_k_ip_2 /= k;
    float mmr_2 = lambda*mean_k_ip_2 - c*(1-lambda)*max_pair_ip_2;
    
    // choose the larger f(S) and update S
    if (mmr_1 >= mmr_2) {
        mean_k_ip = mean_k_ip_1; max_pair_ip = max_pair_ip_1; mmr = mmr_1;
        for (int i = 0; i < k; ++i) res[i] = res_1[i];
    } else {
        mean_k_ip = mean_k_ip_2; max_pair_ip = max_pair_ip_2; mmr = mmr_2;
        for (int i = 0; i < k; ++i) res[i] = res_2[i];
    }
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    
    // release space
    delete[] added; delete[] ip; delete list;
    delete[] res_1; delete[] res_2; delete[] max_ip_1; delete[] max_ip_2; delete[] res;
    return result;
}

} // end namespace ip