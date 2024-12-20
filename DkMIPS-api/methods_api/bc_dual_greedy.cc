#include "bc_dual_greedy.h"

namespace ip {

// -----------------------------------------------------------------------------
BC_Dual::BC_Dual(                   // constructor
    int   n,                            // item cardinality
    int   d,                            // dimensionality
    int   d2,                           // dimensionality for 2nd space
    const float *item_set,              // item set by MF
    const float *i2v_set)               // i2v  set by item2vec
    : n_(n), d_(d), d2_(d2), item_set_(item_set), i2v_set_(i2v_set),
    tree_(nullptr)
{
    // init bc-tree
    tree_ = new BC_Tree(n, d, d2_, item_set, i2v_set);
}

// -----------------------------------------------------------------------------
BC_Dual::~BC_Dual()                 // destructor
{
    if (tree_ != nullptr) { delete tree_; tree_ = nullptr; }
}

// -----------------------------------------------------------------------------
int* BC_Dual::dkmips_avg(           // diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *check = new bool[n_];    // whether each p is checked the ip
    bool  *added = new bool[n_];    // whether each p is added to S
    float *ip    = new float[n_];   // ip between each p \in P and q
    float norm_q = sqrt(calc_inner_product(d_, query, query)); // l2-norm of q
    
    memset(check, false, sizeof(bool)*n_);
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];
    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1    = new Result[k];
    Result *res_2    = new Result[k];
    float  *items_1  = new float[k*d_]; // result items for S1
    float  *items_2  = new float[k*d_]; // result items for S2
    Result *sum_ip_1 = new Result[n_];  // sum of ip of p \in P and S1
    Result *sum_ip_2 = new Result[n_];  // sum of ip of p \in P and S2
    
    for (int i = 0; i < n_; ++i) { 
        sum_ip_1[i].id_ = 0; sum_ip_1[i].key_ = 0.0f;
        sum_ip_2[i].id_ = 0; sum_ip_2[i].key_ = 0.0f;
    }
    
    MaxK_List *list = new MaxK_List(2);
    tree_->mips(norm_q, query, list, check, ip);
    
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
        
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    copy_item(max_id_1, d_, item_set_, items_1);
    float mean_k_ip_1 = tau_1, mean_pair_ip_1 = 0.0f;
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    copy_item(max_id_2, d_, item_set_, items_2);
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
            tree_->diverse_mips(cnt_1, lambda, factor, norm_q, added, query, 
                items_1, max_id_1, tau_1, local_sum_1, check, ip, sum_ip_1);
        }
        
        // evaluate the score f(p, S2) for each p \in P\(S1 and S2)
        local_sum_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1;
        if (cnt_2 < k) {
            tree_->diverse_mips(cnt_2, lambda, factor, norm_q, added, query, 
                items_2, max_id_2, tau_2, local_sum_2, check, ip, sum_ip_2);
        }
        
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1];
            mean_k_ip_1 += ip[max_id_1]; mean_pair_ip_1 += local_sum_1;
            copy_item(max_id_1, d_, item_set_, items_1+cnt_1*d_);
            added[max_id_1] = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2];
            mean_k_ip_2 += ip[max_id_2]; mean_pair_ip_2 += local_sum_2;
            copy_item(max_id_2, d_, item_set_, items_2+cnt_2*d_);
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
    delete[] check; delete[] added; delete[] ip; delete list;
    delete[] res_1; delete[] sum_ip_1; delete[] items_1;
    delete[] res_2; delete[] sum_ip_2; delete[] items_2;
    delete[] res;
    return result;
}

// -----------------------------------------------------------------------------
void BC_Dual::copy_item(            // copy item with max id to result items
    int   max_id,                       // max id
    int   d,                            // item dimension from MF/item2vec
    const float* items,                 // items from MF/item2vec
    float *dest)                        // destination of result items (return)
{
    const float *src = items + (u64) max_id*d;
    std::copy(src, src+d, dest);
}

// -----------------------------------------------------------------------------
int* BC_Dual::dkmips_max(           // diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *check = new bool[n_];   // whether each p is checked the ip
    bool  *added = new bool[n_];   // whether each p is added to S
    float *ip    = new float[n_];  // ip between each p \in P and q
    float norm_q = sqrt(calc_inner_product(d_, query, query)); // l2-norm of q
    
    memset(check, false, sizeof(bool)*n_);
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1    = new Result[k];
    Result *res_2    = new Result[k];
    float  *items_1  = new float[k*d_]; // result items for S1
    float  *items_2  = new float[k*d_]; // result items for S2
    Result *max_ip_1 = new Result[n_];  // max ip of p \in P and S1
    Result *max_ip_2 = new Result[n_];  // max ip of p \in P and S2
    
    for (int i = 0; i < n_; ++i) { 
        max_ip_1[i].id_ = 0; max_ip_1[i].key_ = MINREAL; 
        max_ip_2[i].id_ = 0; max_ip_2[i].key_ = MINREAL; 
    }
    
    MaxK_List *list = new MaxK_List(2);
    tree_->mips(norm_q, query, list, check, ip);
    
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
    
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    copy_item(max_id_1, d_, item_set_, items_1);
    float mean_k_ip_1 = tau_1, max_pair_ip_1 = 0.0f; // init 0 as no pair
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    copy_item(max_id_2, d_, item_set_, items_2);
    float mean_k_ip_2 = tau_2, max_pair_ip_2 = 0.0f; // init 0 as no pair
    
    // -------------------------------------------------------------------------
    //  step 2: add j \in [2,k] item p^j into S1 and S2
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    float local_max_1 = -1.0f, local_max_2 = -1.0f;
    int   cnt_1 = 1, cnt_2 = 1;
    
    while (cnt_1 < k || cnt_2 < k) {
        // evaluate the score f(p, S1) for each p \in P\(S1 and S2)
        local_max_1 = -1.0f; tau_1 = MINREAL; max_id_1 = -1;
        if (cnt_1 < k) {
            tree_->diverse_mips(cnt_1, f1, f2, max_pair_ip_1, norm_q, added, 
                query, items_1, max_id_1, tau_1, local_max_1, check, ip, 
                max_ip_1);
        }
        
        // evaluate the score f(p, S2) for each p \in P\(S1 and S2)
        local_max_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1;
        if (cnt_2 < k) {
            tree_->diverse_mips(cnt_2, f1, f2, max_pair_ip_2, norm_q, added, 
                query, items_2, max_id_2, tau_2, local_max_2, check, ip, 
                max_ip_2);
        }
        
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1]; 
            mean_k_ip_1 += ip[max_id_1]; max_pair_ip_1 = local_max_1;
            copy_item(max_id_1, d_, item_set_, items_1+cnt_1*d_);
            added[max_id_1] = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2]; 
            mean_k_ip_2 += ip[max_id_2]; max_pair_ip_2 = local_max_2;
            copy_item(max_id_2, d_, item_set_, items_2+cnt_2*d_);
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
    delete[] check; delete[] added; delete[] ip; delete list;
    delete[] res_1; delete[] max_ip_1; delete[] items_1;
    delete[] res_2; delete[] max_ip_2; delete[] items_2;
    delete[] res;
    return result;
}

// -----------------------------------------------------------------------------
//  the following four methods consider two spaces
// -----------------------------------------------------------------------------
int* BC_Dual::dkmips_avg_i2v(       // diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *check = new bool[n_];    // whether each p is checked the ip
    bool  *added = new bool[n_];    // whether each p is added to S
    float *ip    = new float[n_];   // ip between each p \in P and q
    float norm_q = sqrt(calc_inner_product(d_, query, query)); // l2-norm of q
    
    memset(check, false, sizeof(bool)*n_);
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];
    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1    = new Result[k];
    Result *res_2    = new Result[k];
    float  *items_1  = new float[k*d2_]; // result items for S1
    float  *items_2  = new float[k*d2_]; // result items for S2
    Result *sum_ip_1 = new Result[n_];  // sum of ip of p \in P and S1
    Result *sum_ip_2 = new Result[n_];  // sum of ip of p \in P and S2
    
    for (int i = 0; i < n_; ++i) { 
        sum_ip_1[i].id_ = 0; sum_ip_1[i].key_ = 0.0f;
        sum_ip_2[i].id_ = 0; sum_ip_2[i].key_ = 0.0f;
    }
    
    MaxK_List *list = new MaxK_List(2);
    tree_->mips(norm_q, query, list, check, ip);
    
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
        
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    copy_item(max_id_1, d2_, i2v_set_, items_1);
    float mean_k_ip_1 = tau_1, mean_pair_ip_1 = 0.0f;
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    copy_item(max_id_2, d2_, i2v_set_, items_2);
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
            tree_->diverse_mips_i2v(cnt_1, lambda, factor, norm_q, added, query, 
                items_1, max_id_1, tau_1, local_sum_1, check, ip, sum_ip_1);
        }
        
        // evaluate the score f(p, S2) for each p \in P\(S1 and S2)
        local_sum_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1;
        if (cnt_2 < k) {
            tree_->diverse_mips_i2v(cnt_2, lambda, factor, norm_q, added, query, 
                items_2, max_id_2, tau_2, local_sum_2, check, ip, sum_ip_2);
        }
        
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1];
            mean_k_ip_1 += ip[max_id_1]; mean_pair_ip_1 += local_sum_1;
            copy_item(max_id_1, d2_, i2v_set_, items_1+cnt_1*d2_);
            added[max_id_1] = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2];
            mean_k_ip_2 += ip[max_id_2]; mean_pair_ip_2 += local_sum_2;
            copy_item(max_id_2, d2_, i2v_set_, items_2+cnt_2*d2_);
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
    delete[] check; delete[] added; delete[] ip; delete list;
    delete[] res_1; delete[] sum_ip_1; delete[] items_1;
    delete[] res_2; delete[] sum_ip_2; delete[] items_2;
    delete[] res;
    return result;
}

// -----------------------------------------------------------------------------
int* BC_Dual::dkmips_max_i2v(       // diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    bool  *check = new bool[n_];   // whether each p is checked the ip
    bool  *added = new bool[n_];   // whether each p is added to S
    float *ip    = new float[n_];  // ip between each p \in P and q
    float norm_q = sqrt(calc_inner_product(d_, query, query)); // l2-norm of q
    
    memset(check, false, sizeof(bool)*n_);
    memset(added, false, sizeof(bool)*n_);
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    // -------------------------------------------------------------------------
    //  step 1: init two result sets S1 and S2 by the the first two items with 
    //  the maximum inner products of query
    // -------------------------------------------------------------------------
    Result *res_1    = new Result[k];
    Result *res_2    = new Result[k];
    float  *items_1  = new float[k*d2_]; // result items for S1
    float  *items_2  = new float[k*d2_]; // result items for S2
    Result *max_ip_1 = new Result[n_];  // max ip of p \in P and S1
    Result *max_ip_2 = new Result[n_];  // max ip of p \in P and S2
    
    for (int i = 0; i < n_; ++i) { 
        max_ip_1[i].id_ = 0; max_ip_1[i].key_ = MINREAL; 
        max_ip_2[i].id_ = 0; max_ip_2[i].key_ = MINREAL; 
    }
    
    MaxK_List *list = new MaxK_List(2);
    tree_->mips(norm_q, query, list, check, ip);
    
    float tau_1    = list->ith_key(0), tau_2    = list->ith_key(1);
    int   max_id_1 = list->ith_id(0),  max_id_2 = list->ith_id(1);
    
    // update S1
    res_1[0].id_ = max_id_1; res_1[0].key_ = tau_1; added[max_id_1] = true;
    copy_item(max_id_1, d2_, i2v_set_, items_1);
    float mean_k_ip_1 = tau_1, max_pair_ip_1 = 0.0f; // init 0 as no pair
    
    // update S2
    res_2[0].id_ = max_id_2; res_2[0].key_ = tau_2; added[max_id_2] = true;
    copy_item(max_id_2, d2_, i2v_set_, items_2);
    float mean_k_ip_2 = tau_2, max_pair_ip_2 = 0.0f; // init 0 as no pair
    
    // -------------------------------------------------------------------------
    //  step 2: add j \in [2,k] item p^j into S1 and S2
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    float local_max_1 = -1.0f, local_max_2 = -1.0f;
    int   cnt_1 = 1, cnt_2 = 1;
    
    while (cnt_1 < k || cnt_2 < k) {
        // evaluate the score f(p, S1) for each p \in P\(S1 and S2)
        local_max_1 = -1.0f; tau_1 = MINREAL; max_id_1 = -1;
        if (cnt_1 < k) {
            tree_->diverse_mips_i2v(cnt_1, f1, f2, max_pair_ip_1, norm_q, added, 
                query, items_1, max_id_1, tau_1, local_max_1, check, ip, 
                max_ip_1);
        }
        
        // evaluate the score f(p, S2) for each p \in P\(S1 and S2)
        local_max_2 = -1.0f; tau_2 = MINREAL; max_id_2 = -1;
        if (cnt_2 < k) {
            tree_->diverse_mips_i2v(cnt_2, f1, f2, max_pair_ip_2, norm_q, added, 
                query, items_2, max_id_2, tau_2, local_max_2, check, ip, 
                max_ip_2);
        }
        
        // add item with the largest score into S1 or S2
        if (tau_1 >= tau_2) {
            res_1[cnt_1].id_ = max_id_1; res_1[cnt_1].key_ = ip[max_id_1]; 
            mean_k_ip_1 += ip[max_id_1]; max_pair_ip_1 = local_max_1;
            copy_item(max_id_1, d2_, i2v_set_, items_1+cnt_1*d2_);
            added[max_id_1] = true; ++cnt_1;
        } else {
            res_2[cnt_2].id_ = max_id_2; res_2[cnt_2].key_ = ip[max_id_2]; 
            mean_k_ip_2 += ip[max_id_2]; max_pair_ip_2 = local_max_2;
            copy_item(max_id_2, d2_, i2v_set_, items_2+cnt_2*d2_);
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
    delete[] check; delete[] added; delete[] ip; delete list;
    delete[] res_1; delete[] max_ip_1; delete[] items_1;
    delete[] res_2; delete[] max_ip_2; delete[] items_2;
    delete[] res;
    return result;
}

} // end namespace ip