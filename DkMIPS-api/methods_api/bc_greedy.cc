#include "bc_greedy.h"

namespace ip {

// -----------------------------------------------------------------------------
BC_Greedy::BC_Greedy(               // constructor
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
BC_Greedy::~BC_Greedy()             // destructor
{
    if (tree_ != nullptr) { delete tree_; tree_ = nullptr; }
}

// -----------------------------------------------------------------------------
int* BC_Greedy::dkmips_avg(         // diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool   *check  = new bool[n_];      // whether each p is checked the ip
    bool   *added  = new bool[n_];      // whether each p is added to S
    float  *ip     = new float[n_];     // ip between each p \in P and q
    float  *items  = new float[k*d_];   // result items
    Result *sum_ip = new Result[n_];    // sum of ip of p \in P and S
    
    memset(check,  false, sizeof(bool)*n_);
    memset(added,  false, sizeof(bool)*n_);

    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];

    for (int i = 0; i < n_; ++i) { sum_ip[i].id_=0; sum_ip[i].key_=0.0f; }
    
    float norm_q = sqrt(calc_inner_product(d_, query, query)); // l2-norm of q
    float tau    = MINREAL;         // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    
    tree_->mips(norm_q, query, max_id, tau, check, ip);
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    copy_item(max_id, d_, item_set_, items);
    mean_k_ip = tau; mean_pair_ip = 0.0;
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float factor = 2*c*(1-lambda)/(k-1);
    for (int j = 1; j < k; ++j) {
        // evaluate the score f(p,S) for each p \in P
        float local_sum = -1.0f; tau = MINREAL; max_id = -1;
        tree_->diverse_mips(j, lambda, factor, norm_q, added, query, items, 
            max_id, tau, local_sum, check, ip, sum_ip);
        
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true;
        copy_item(max_id, d_, item_set_, items+j*d_);
        mean_k_ip += ip[max_id]; mean_pair_ip += local_sum;
    }
    mean_k_ip /= k;
    mean_pair_ip = mean_pair_ip*2/(k*(k-1));
    mmr = lambda*mean_k_ip - c*(1-lambda)*mean_pair_ip;
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    
    // release space
    delete[] check;  delete[] added; delete[] ip; 
    delete[] sum_ip; delete[] items;
    delete[] res;
    return result;
}

// -----------------------------------------------------------------------------
void BC_Greedy::copy_item(          // copy item with max id to result items
    int   max_id,                       // max id
    int   d,                            // item dimension from MF/item2vec
    const float* items,                 // items from MF/item2vec
    float *dest)                        // destination of result items (return)
{
    const float *src = items + (u64) max_id*d;
    std::copy(src, src+d, dest);
}

// -----------------------------------------------------------------------------
int* BC_Greedy::dkmips_max(         // diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool   *check  = new bool[n_];      // whether each p is checked the ip
    bool   *added  = new bool[n_];      // whether each p is added to S
    float  *ip     = new float[n_];     // ip between each p \in P and q
    float  *items  = new float[k*d_];   // result items
    Result *max_ip = new Result[n_];    // max ip of p \in P and S
    
    memset(check,  false, sizeof(bool)*n_);
    memset(added,  false, sizeof(bool)*n_);
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    
    for (int i = 0; i < n_; ++i) { max_ip[i].id_=0; max_ip[i].key_=MINREAL; }
    
    float norm_q = sqrt(calc_inner_product(d_, query, query)); // l2-norm of q
    float tau    = MINREAL;         // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    
    tree_->mips(norm_q, query, max_id, tau, check, ip);
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    copy_item(max_id, d_, item_set_, items);
    mean_k_ip = tau; max_pair_ip = 0; // init 0 as there is no pair
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    for (int j = 1; j < k; ++j) {
        // evaluate the score f(p,S) for each p \in P
        float local_max = -1.0f; tau = MINREAL; max_id = -1;
        tree_->diverse_mips(j, f1, f2, max_pair_ip, norm_q, added, query, 
            items, max_id, tau, local_max, check, ip, max_ip);
        
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true;
        copy_item(max_id, d_, item_set_, items+j*d_);
        mean_k_ip += ip[max_id]; max_pair_ip = local_max;
    }
    mean_k_ip /= k;
    mmr = lambda*mean_k_ip - c*(1-lambda)*max_pair_ip;
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    
    // release space
    delete[] check;  delete[] added; delete[] ip; 
    delete[] max_ip; delete[] items;
    delete[] res;
    return result;
}




// -----------------------------------------------------------------------------
//  the following two methods consider two spaces
// -----------------------------------------------------------------------------
int* BC_Greedy::dkmips_avg_i2v(     // diversity-aware k-mips (avg)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool   *check  = new bool[n_];      // whether each p is checked the ip
    bool   *added  = new bool[n_];      // whether each p is added to S
    float  *ip     = new float[n_];     // ip between each p \in P and q
    float  *items  = new float[k*d2_];  // result items
    Result *sum_ip = new Result[n_];    // sum of ip of p \in P and S
    
    memset(check,  false, sizeof(bool)*n_);
    memset(added,  false, sizeof(bool)*n_);
    float mmr, mean_k_ip, mean_pair_ip;
    Result *res = new Result[k];
    
    for (int i = 0; i < n_; ++i) { sum_ip[i].id_=0; sum_ip[i].key_=0.0f; }
    
    float norm_q = sqrt(calc_inner_product(d_, query, query)); // l2-norm of q
    float tau    = MINREAL;         // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    
    tree_->mips(norm_q, query, max_id, tau, check, ip);
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    copy_item(max_id, d2_, i2v_set_, items);
    mean_k_ip = tau; mean_pair_ip = 0.0;
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float factor = 2*c*(1-lambda)/(k-1);
    for (int j = 1; j < k; ++j) {
        // evaluate the score f(p,S) for each p \in P
        float local_sum = -1.0f; tau = MINREAL; max_id = -1;
        tree_->diverse_mips_i2v(j, lambda, factor, norm_q, added, query, items, 
            max_id, tau, local_sum, check, ip, sum_ip);
        
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true;
        copy_item(max_id, d2_, i2v_set_, items+j*d2_);
        mean_k_ip += ip[max_id]; mean_pair_ip += local_sum;
    }
    mean_k_ip /= k;
    mean_pair_ip = mean_pair_ip*2/(k*(k-1));
    mmr = lambda*mean_k_ip - c*(1-lambda)*mean_pair_ip;
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    
    // release space
    delete[] check;  delete[] added; delete[] ip; 
    delete[] sum_ip; delete[] items;
    delete[] res;
    return result;
}

// -----------------------------------------------------------------------------
int* BC_Greedy::dkmips_max_i2v(     // diversity-aware k-mips (max)
    int    k,                           // top-k value
    float  lambda,                      // balance factor
    float  c,                           // scale factor
    const  float *query)                // query (user) vector
{
    // -------------------------------------------------------------------------
    //  step 1: init the result set S by the item p* with the MIPS of query
    // -------------------------------------------------------------------------
    bool   *check  = new bool[n_];      // whether each p is checked the ip
    bool   *added  = new bool[n_];      // whether each p is added to S
    float  *ip     = new float[n_];     // ip between each p \in P and q
    float  *items  = new float[k*d2_];  // result items
    Result *max_ip = new Result[n_];    // max ip of p \in P and S
    
    memset(check,  false, sizeof(bool)*n_);
    memset(added,  false, sizeof(bool)*n_);
    float mmr, mean_k_ip, max_pair_ip;
    Result *res = new Result[k];
    
    for (int i = 0; i < n_; ++i) { max_ip[i].id_=0; max_ip[i].key_=MINREAL; }
    
    float norm_q = sqrt(calc_inner_product(d_, query, query)); // l2-norm of q
    float tau    = MINREAL;         // max ip between P and q
    int   max_id = -1;              // item id for the max ip (or max score)
    
    tree_->mips(norm_q, query, max_id, tau, check, ip);
    res[0].id_ = max_id; res[0].key_ = tau; added[max_id] = true;
    copy_item(max_id, d2_, i2v_set_, items);
    mean_k_ip = tau; max_pair_ip = 0; // init 0 as there is no pair
    
    // -------------------------------------------------------------------------
    //  step 2 & 3: add j \in [2,k] item p^j into S
    // -------------------------------------------------------------------------
    float f1 = lambda/k, f2 = c*(1-lambda);
    for (int j = 1; j < k; ++j) {
        // evaluate the score f(p,S) for each p \in P
        float local_max = -1.0f; tau = MINREAL; max_id = -1;
        tree_->diverse_mips_i2v(j, f1, f2, max_pair_ip, norm_q, added, query, 
            items, max_id, tau, local_max, check, ip, max_ip);
        
        // add the item id with the largest score into S
        res[j].id_ = max_id; res[j].key_ = ip[max_id]; added[max_id] = true;
        copy_item(max_id, d2_, i2v_set_, items+j*d2_);
        mean_k_ip += ip[max_id]; max_pair_ip = local_max;
    }
    mean_k_ip /= k;
    mmr = lambda*mean_k_ip - c*(1-lambda)*max_pair_ip;
    
    int* result = new int[k];
    for (int i = 0; i < k; ++i) {
        result[i] = res[i].id_;
    }
    
    // release space
    delete[] check;  delete[] added; delete[] ip; 
    delete[] max_ip; delete[] items;
    delete[] res;
    return result;
}

} // end namespace ip