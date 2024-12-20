#include "bc_tree.h"

namespace ip {

// -----------------------------------------------------------------------------
//  BC_Node: leaf node and internal node of BC_Tree
// -----------------------------------------------------------------------------
BC_Node::BC_Node(                   // constructor
    int   n,                            // item cardinality
    int   d,                            // dimensionality
    int   d2,                           // dimensionality for 2nd space
    bool  is_leaf,                      // is leaf node
    BC_Node *lc,                        // left  child
    BC_Node *rc,                        // right child
    int   *index,                       // item  index (allow modify)
    const float *item_set,              // item set by MF
    const float *i2v_set)               // i2v  set by item2vec
    : n_(n), d_(d), d2_(d2), radius_(-1.0f), center_(nullptr), lc_(lc), rc_(rc),
    index_(index), item_set_(nullptr), i2v_set_(i2v_set), norm_c_(-1.0f), 
    r_x_(nullptr), x_cos_(nullptr), x_sin_(nullptr)
{
    center_ = new float[d];
    if (is_leaf) {
        // calc the center and its l2-norm
        calc_centroid(n, d, index, item_set, center_);
        norm_c_ = sqrt(calc_inner_product(d, center_, center_));
        
        // calc the radius
        Result *result = new Result[n];
        for (int i = 0; i < n; ++i) {
            const float *item = item_set + (u64) index[i]*d;
            result[i].id_  = index[i];
            result[i].key_ = calc_l2_dist(d, item, center_);
        }
        qsort(result, n, sizeof(Result), ResultCompDesc);
        radius_ = result[0].key_;
        
        // re-order index_ & item_set_ and calc r_x_, x_cos_, & x_sin_
        item_set_ = new float[(u64) n*d];
        r_x_ = new float[n]; x_cos_ = new float[n]; x_sin_ = new float[n]; 
        
        for (int i = 0; i < n; ++i) {
            index_[i] = result[i].id_;
            r_x_[i]   = result[i].key_;
            
            const float *item = item_set + (u64) index_[i]*d;
            float norm2 = calc_inner_product(d, item, item);
            float ip    = calc_inner_product(d, item, center_);
            
            x_cos_[i] = ip / norm_c_;
            x_sin_[i] = sqrt(norm2 - SQR(x_cos_[i]));
            std::copy(item, item+d, item_set_+(u64)i*d);
        }
        delete[] result;
    }
    else {
        // calc the center based on its two leaves
        int   ln = lc_->n_, rn = rc_->n_; // assert(ln + rn == n);
        float *l_ctrd = lc_->center_, *r_ctrd = rc_->center_;
        for (int i = 0; i < d; ++i) {
            center_[i] = (ln*l_ctrd[i] + rn*r_ctrd[i]) / n;
        }
        
        // calc the radius
        radius_ = -1.0f;
        for (int i = 0; i < n; ++i) {
            const float *item = item_set + (u64) index[i]*d;
            float dist = calc_l2_sqr(d, item, center_);
            if (dist > radius_) radius_ = dist;
        }
        radius_ = sqrt(radius_);
    }
}

// -----------------------------------------------------------------------------
BC_Node::~BC_Node()                 // destructor
{
    if (lc_ != nullptr) { delete lc_; lc_ = nullptr; }
    if (rc_ != nullptr) { delete rc_; rc_ = nullptr; }
    
    if (center_   != nullptr) { delete[] center_;   center_   = nullptr; }
    if (item_set_ != nullptr) { delete[] item_set_; item_set_ = nullptr; }
    if (r_x_      != nullptr) { delete[] r_x_;      r_x_      = nullptr; }
    if (x_cos_    != nullptr) { delete[] x_cos_;    x_cos_    = nullptr; }
    if (x_sin_    != nullptr) { delete[] x_sin_;    x_sin_    = nullptr; }
}

// -----------------------------------------------------------------------------
void BC_Node::mips(                 // mips on bc-node (for avg & max)
    float cq_ip,                        // inner product for center & query
    float norm_q,                       // l2-norm of query
    const float *query,                 // query (user) vector
    int   &max_id,                      // max item id (return)
    float &tau,                         // max ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip)                          // ip between item and query (return)
{
    // early stop: check ball upper bound
    float ball_ub = cq_ip + norm_q*radius_;
    if (ball_ub <= tau) return;
    
    // traversal the tree
    if (item_set_ != nullptr) { // leaf node
        linear_scan(cq_ip, norm_q, query, max_id, tau, check, ip);
    }
    else { // internal node
        // center preference 
        float lc_ip = calc_inner_product(d_, lc_->center_, query);
        float rc_ip = (cq_ip*n_ - lc_ip*lc_->n_) / rc_->n_; 
        if (lc_ip > rc_ip) {
            lc_->mips(lc_ip, norm_q, query, max_id, tau, check, ip);
            rc_->mips(rc_ip, norm_q, query, max_id, tau, check, ip);
        } else {
            rc_->mips(rc_ip, norm_q, query, max_id, tau, check, ip);
            lc_->mips(lc_ip, norm_q, query, max_id, tau, check, ip);
        }
    }
}

// -----------------------------------------------------------------------------
void BC_Node::linear_scan(          // linear scan (for avg & max)
    float cq_ip,                        // inner product for center & query
    float norm_q,                       // l2-norm of query
    const float *query,                 // query (user) vector
    int   &max_id,                      // max item id (return)
    float &tau,                         // max ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip)                          // ip between item and query (return)
{
    float q_cos = cq_ip / norm_c_;
    float q_sin = sqrt(norm_q*norm_q - q_cos*q_cos);
    
    for (int i = 0; i < n_; ++i) {
        float ub = cq_ip + norm_q*r_x_[i]; // ball upper bound
        if (ub <= tau) return;
        
        ub = q_cos*x_cos_[i] + q_sin*x_sin_[i]; // cone upper bound
        if (ub <= tau) continue;
        
        // compute the inner product
        int id = index_[i];
        const float *item = item_set_ + (u64) i*d_;
        ip[id] = calc_inner_product(d_, item, query);
        check[id] = true;
        if (ip[id] > tau) { tau = ip[id]; max_id = id; }
    }
}

// -----------------------------------------------------------------------------
void BC_Node::mips(                 // mips on bc-node (for avg & max)
    float cq_ip,                        // inner product for center & query
    float norm_q,                       // l2-norm of query
    const float *query,                 // query (user) vector
    MaxK_List *list,                    // top-k results (return)
    bool  *check,                       // ip is checked? (return)
    float *ip)                          // ip between item and query (return)
{
    // early stop: check ball upper bound
    float ball_ub = cq_ip + norm_q*radius_;
    if (ball_ub <= list->min_key()) return;
    
    // traversal the tree
    if (item_set_ != nullptr) { // leaf node
        linear_scan(cq_ip, norm_q, query, list, check, ip);
    }
    else { // internal node
        // center preference 
        float lc_ip = calc_inner_product(d_, lc_->center_, query);
        float rc_ip = (cq_ip*n_ - lc_ip*lc_->n_) / rc_->n_; 
        if (lc_ip > rc_ip) {
            lc_->mips(lc_ip, norm_q, query, list, check, ip);
            rc_->mips(rc_ip, norm_q, query, list, check, ip);
        } else {
            rc_->mips(rc_ip, norm_q, query, list, check, ip);
            lc_->mips(lc_ip, norm_q, query, list, check, ip);
        }
    }
}

// -----------------------------------------------------------------------------
void BC_Node::linear_scan(          // linear scan (for avg & max)
    float cq_ip,                        // inner product for center & query
    float norm_q,                       // l2-norm of query
    const float *query,                 // query (user) vector
    MaxK_List *list,                    // top-k results (return)
    bool  *check,                       // ip is checked? (return)
    float *ip)                          // ip between item and query (return)
{
    float q_cos = cq_ip / norm_c_;
    float q_sin = sqrt(norm_q*norm_q - q_cos*q_cos);
    
    float tau = list->min_key();
    for (int i = 0; i < n_; ++i) {
        float ub = cq_ip + norm_q*r_x_[i]; // ball upper bound
        if (ub <= tau) return;
        
        ub = q_cos*x_cos_[i] + q_sin*x_sin_[i]; // cone upper bound
        if (ub <= tau) continue;
        
        // compute the actual inner product
        int id = index_[i];
        const float *item = item_set_ + (u64) i*d_;
        ip[id] = calc_inner_product(d_, item, query);
        check[id] = true;
        tau = list->insert(ip[id], id);
    }
}

// -----------------------------------------------------------------------------
void BC_Node::diverse_mips(         // diverse mips on bc-node (for avg)
    int   num,                          // num of id in result set
    float cq_ip,                        // inner product for center & query
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item 
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_sum,                   // local max sum of ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *sum_ip)                     // sum of ip (return)
{
    // early stop: check ball upper bound
    float ball_ub = f1*(cq_ip + norm_q*radius_);
    if (ball_ub <= tau) return;
    
    // traversal the tree
    if (item_set_ != nullptr) { // leaf node
        diverse_linear_scan(num, cq_ip, f1, f2, norm_q, added, query, items, 
            max_id, tau, local_sum, check, ip, sum_ip);
    }
    else { // internal node
        // center preference 
        float lc_ip = calc_inner_product(d_, lc_->center_, query);
        float rc_ip = (cq_ip*n_ - lc_ip*lc_->n_) / rc_->n_; 
        if (lc_ip > rc_ip) {
            lc_->diverse_mips(num, lc_ip, f1, f2, norm_q, added, query, items, 
                max_id, tau, local_sum, check, ip, sum_ip);
            rc_->diverse_mips(num, rc_ip, f1, f2, norm_q, added, query, items, 
                max_id, tau, local_sum, check, ip, sum_ip);
        } else {
            rc_->diverse_mips(num, rc_ip, f1, f2, norm_q, added, query, items, 
                max_id, tau, local_sum, check, ip, sum_ip);
            lc_->diverse_mips(num, lc_ip, f1, f2, norm_q, added, query, items, 
                max_id, tau, local_sum, check, ip, sum_ip);
        }
    }
}

// -----------------------------------------------------------------------------
void BC_Node::diverse_linear_scan(  // diverse linear scan (for avg)
    int   num,                          // num of id in result set
    float cq_ip,                        // inner product for center & query
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item 
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_sum,                   // local max sum of ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *sum_ip)                     // sum of ip (return)
{
    float q_cos = cq_ip / norm_c_;
    float q_sin = sqrt(norm_q*norm_q - q_cos*q_cos);
    
    for (int i = 0; i < n_; ++i) {
        int id = index_[i];
        if (added[id]) continue; // skip the added item
        
        float ub = f1*(cq_ip + norm_q*r_x_[i]); // ball upper bound
        if (ub <= tau) return; 
        
        // calc the score f(p, S) and update max_id, tau, & local_sum
        const float *target = item_set_ + (u64) i*d_;
        if (!check[id]) {
            ub = f1*(q_cos*x_cos_[i] + q_sin*x_sin_[i]); // cone upper bound
            if (ub <= tau) continue;
        
            ip[id] = calc_inner_product(d_, target, query); 
            check[id] = true;
        }
        if (f1*ip[id] <= tau) continue; // ip upper bound
        
        float this_sum_ip = calc_sum_ip(num, id, d_, target, items, sum_ip);
        float score = f1*ip[id] - f2*this_sum_ip;
        if (score > tau) { tau=score; max_id=id; local_sum=this_sum_ip; }
    }
}

// -----------------------------------------------------------------------------
void BC_Node::diverse_mips(         // diverse mips on bc-node (for max)
    int   num,                          // num of id in result set
    float cq_ip,                        // inner product for center & query
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float max_pair_ip,                  // max pair-wise ip
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_max,                   // local max ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *max_ip)                     // max ip (return)
{
    // early stop: check ball upper bound
    float ball_ub = f1*(cq_ip + norm_q*radius_);
    if (ball_ub <= tau) return;
    
    // traversal the tree
    if (item_set_ != nullptr) { // leaf node
        diverse_linear_scan(num, cq_ip, f1, f2, max_pair_ip, norm_q, added, 
            query, items, max_id, tau, local_max, check, ip, max_ip);
    }
    else { // internal node
        // center preference 
        float lc_ip = calc_inner_product(d_, lc_->center_, query);
        float rc_ip = (cq_ip*n_ - lc_ip*lc_->n_) / rc_->n_; 
        if (lc_ip > rc_ip) {
            lc_->diverse_mips(num, lc_ip, f1, f2, max_pair_ip, norm_q, added, 
                query, items, max_id, tau, local_max, check, ip, max_ip);
            rc_->diverse_mips(num, rc_ip, f1, f2, max_pair_ip, norm_q, added, 
                query, items, max_id, tau, local_max, check, ip, max_ip);
        } else {
            rc_->diverse_mips(num, rc_ip, f1, f2, max_pair_ip, norm_q, added, 
                query, items, max_id, tau, local_max, check, ip, max_ip);
            lc_->diverse_mips(num, lc_ip, f1, f2, max_pair_ip, norm_q, added, 
                query, items, max_id, tau, local_max, check, ip, max_ip);
        }
    }
}

// -----------------------------------------------------------------------------
void BC_Node::diverse_linear_scan(  // diverse linear scan (for max)
    int   num,                          // num of id in result set
    float cq_ip,                        // inner product for center & query
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float max_pair_ip,                  // max pair-wise ip
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_max,                   // local max ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *max_ip)                     // max ip (return)
{
    float q_cos = cq_ip / norm_c_;
    float q_sin = sqrt(norm_q*norm_q - q_cos*q_cos);
    
    for (int i = 0; i < n_; ++i) {
        int id = index_[i];
        if (added[id]) continue; // skip the added item
        
        float ub = f1*(cq_ip + norm_q*r_x_[i]); // ball upper bound
        if (ub <= tau) return; 
        
        // calc the score f(p, S) and update max_id, tau, & local_max
        const float *target = item_set_ + (u64) i*d_;
        if (!check[id]) {
            ub = f1*(q_cos*x_cos_[i] + q_sin*x_sin_[i]); // cone upper bound
            if (ub <= tau) continue; 
            
            ip[id] = calc_inner_product(d_, target, query); 
            check[id] = true;
        }
        if (f1*ip[id] <= tau) continue; // ip upper bound
        
        float this_max = find_max_ip(num, id, d_, target, items, max_ip);
        this_max = std::max(max_pair_ip, this_max);
        float score = f1*ip[id] - f2*(this_max - max_pair_ip);
        if (score > tau) { tau=score; max_id=id; local_max=this_max; }
    }
}

// -----------------------------------------------------------------------------
//  the following two methods consider two spaces
// -----------------------------------------------------------------------------
void BC_Node::diverse_mips_i2v(     // diverse mips on bc-node (for avg)
    int   num,                          // num of id in result set
    float cq_ip,                        // inner product for center & query
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item 
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_sum,                   // local max sum of ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *sum_ip)                     // sum of ip (return)
{
    // early stop: check ball upper bound
    float ball_ub = f1*(cq_ip + norm_q*radius_);
    if (ball_ub <= tau) return;
    
    // traversal the tree
    if (item_set_ != nullptr) { // leaf node
        diverse_linear_scan_i2v(num, cq_ip, f1, f2, norm_q, added, query, items, 
            max_id, tau, local_sum, check, ip, sum_ip);
    }
    else { // internal node
        // center preference 
        float lc_ip = calc_inner_product(d_, lc_->center_, query);
        float rc_ip = (cq_ip*n_ - lc_ip*lc_->n_) / rc_->n_; 
        if (lc_ip > rc_ip) {
            lc_->diverse_mips_i2v(num, lc_ip, f1, f2, norm_q, added, query, 
                items, max_id, tau, local_sum, check, ip, sum_ip);
            rc_->diverse_mips_i2v(num, rc_ip, f1, f2, norm_q, added, query, 
                items, max_id, tau, local_sum, check, ip, sum_ip);
        } else {
            rc_->diverse_mips_i2v(num, rc_ip, f1, f2, norm_q, added, query, 
                items, max_id, tau, local_sum, check, ip, sum_ip);
            lc_->diverse_mips_i2v(num, lc_ip, f1, f2, norm_q, added, query, 
                items, max_id, tau, local_sum, check, ip, sum_ip);
        }
    }
}

// -----------------------------------------------------------------------------
void BC_Node::diverse_linear_scan_i2v(// diverse linear scan (for avg)
    int   num,                          // num of id in result set
    float cq_ip,                        // inner product for center & query
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item 
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_sum,                   // local max sum of ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *sum_ip)                     // sum of ip (return)
{
    float q_cos = cq_ip / norm_c_;
    float q_sin = sqrt(norm_q*norm_q - q_cos*q_cos);
    
    for (int i = 0; i < n_; ++i) {
        int id = index_[i];
        if (added[id]) continue; // skip the added item
        
        float ub = f1*(cq_ip + norm_q*r_x_[i]); // ball upper bound
        if (ub <= tau) return; 
        
        // calc the score f(p, S) and update max_id, tau, & local_sum
        const float *target = item_set_ + (u64) i*d_;
        if (!check[id]) {
            ub = f1*(q_cos*x_cos_[i] + q_sin*x_sin_[i]); // cone upper bound
            if (ub <= tau) continue;
        
            ip[id] = calc_inner_product(d_, target, query); 
            check[id] = true;
        }
        if (f1*ip[id] <= tau) continue; // ip upper bound
        
        const float *target2 = i2v_set_ + (u64) id*d2_;
        float this_sum_ip = calc_sum_ip(num, id, d2_, target2, items, sum_ip);
        float score = f1*ip[id] - f2*this_sum_ip;
        if (score > tau) { tau=score; max_id=id; local_sum=this_sum_ip; }
    }
}

// -----------------------------------------------------------------------------
void BC_Node::diverse_mips_i2v(     // diverse mips on bc-node (for max)
    int   num,                          // num of id in result set
    float cq_ip,                        // inner product for center & query
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float max_pair_ip,                  // max pair-wise ip
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_max,                   // local max ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *max_ip)                     // max ip (return)
{
    // early stop: check ball upper bound
    float ball_ub = f1*(cq_ip + norm_q*radius_);
    if (ball_ub <= tau) return;
    
    // traversal the tree
    if (item_set_ != nullptr) { // leaf node
        diverse_linear_scan_i2v(num, cq_ip, f1, f2, max_pair_ip, norm_q, added, 
            query, items, max_id, tau, local_max, check, ip, max_ip);
    }
    else { // internal node
        // center preference 
        float lc_ip = calc_inner_product(d_, lc_->center_, query);
        float rc_ip = (cq_ip*n_ - lc_ip*lc_->n_) / rc_->n_; 
        if (lc_ip > rc_ip) {
            lc_->diverse_mips_i2v(num, lc_ip, f1, f2, max_pair_ip, norm_q, added, 
                query, items, max_id, tau, local_max, check, ip, max_ip);
            rc_->diverse_mips_i2v(num, rc_ip, f1, f2, max_pair_ip, norm_q, added, 
                query, items, max_id, tau, local_max, check, ip, max_ip);
        } else {
            rc_->diverse_mips_i2v(num, rc_ip, f1, f2, max_pair_ip, norm_q, added, 
                query, items, max_id, tau, local_max, check, ip, max_ip);
            lc_->diverse_mips_i2v(num, lc_ip, f1, f2, max_pair_ip, norm_q, added, 
                query, items, max_id, tau, local_max, check, ip, max_ip);
        }
    }
}

// -----------------------------------------------------------------------------
void BC_Node::diverse_linear_scan_i2v(// diverse linear scan (for max)
    int   num,                          // num of id in result set
    float cq_ip,                        // inner product for center & query
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float max_pair_ip,                  // max pair-wise ip
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_max,                   // local max ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *max_ip)                     // max ip (return)
{
    float q_cos = cq_ip / norm_c_;
    float q_sin = sqrt(norm_q*norm_q - q_cos*q_cos);
    
    for (int i = 0; i < n_; ++i) {
        int id = index_[i];
        if (added[id]) continue; // skip the added item
        
        float ub = f1*(cq_ip + norm_q*r_x_[i]); // ball upper bound
        if (ub <= tau) return; 
        
        // calc the score f(p, S) and update max_id, tau, & local_max
        const float *target = item_set_ + (u64) i*d_;
        if (!check[id]) {
            ub = f1*(q_cos*x_cos_[i] + q_sin*x_sin_[i]); // cone upper bound
            if (ub <= tau) continue; 
            
            ip[id] = calc_inner_product(d_, target, query); 
            check[id] = true;
        }
        if (f1*ip[id] <= tau) continue; // ip upper bound
        
        const float *target2 = i2v_set_ + (u64) id*d2_;
        float this_max = find_max_ip(num, id, d2_, target2, items, max_ip);
        this_max = std::max(max_pair_ip, this_max);
        float score = f1*ip[id] - f2*(this_max - max_pair_ip);
        if (score > tau) { tau=score; max_id=id; local_max=this_max; }
    }
}

// -----------------------------------------------------------------------------
//  assistant functions
// -----------------------------------------------------------------------------
float BC_Node::calc_sum_ip(         // calc sum of ip for input id (for avg)
    int   num,                          // num of id in result set
    int   id,                           // input id
    int   d,                            // item dimension
    const float *target,                // target item by input id
    const float *items,                 // result items
    Result *sum_ip)                     // sum of ip (return)
{
    int start = sum_ip[id].id_;
    float sum = sum_ip[id].key_;
    
    for (int i = start; i < num; ++i) {
        const float *item = items + i*d;
        float ip = calc_inner_product(d, target, item);
        sum += ip;
    }
    sum_ip[id].id_  = num;
    sum_ip[id].key_ = sum;
    
    return sum;
}

// -----------------------------------------------------------------------------
float BC_Node::find_max_ip(         // find max ip for input id (for max)
    int   num,                          // num of id in result set
    int   id,                           // input id
    int   d,                            // item dimension
    const float *target,                // target item by input id
    const float *items,                 // result items
    Result *max_ip)                     // max ip (return)
{
    int start = max_ip[id].id_;
    float tau = max_ip[id].key_;
    
    for (int i = start; i < num; ++i) {
        const float *item = items + i*d;
        float ip = calc_inner_product(d, target, item);
        if (ip > tau) tau = ip;
    }
    max_ip[id].id_  = num;
    max_ip[id].key_ = tau;
    
    return tau;
}

// -----------------------------------------------------------------------------
void BC_Node::traversal(            // traversal bc-tree
    std::vector<int> &leaf_size)        // leaf size (return)
{
    if (item_set_ != nullptr) {     // leaf node
        leaf_size.push_back(n_);
    } else {                        // internal node
        lc_->traversal(leaf_size);
        rc_->traversal(leaf_size);
    }
}





// -----------------------------------------------------------------------------
//  BC_Tree maintains a ball structure for internal nodes and a joint structure 
//  of ball & cone for leaf nodes for k-mips
// -----------------------------------------------------------------------------
BC_Tree::BC_Tree(                   // constructor
    int   n,                            // item cardinality
    int   d,                            // dimensionality
    int   d2,                           // dimensionality for 2nd space
    const float *item_set,              // item set by MF
    const float *i2v_set)               // i2v  set by item2vec
    : n_(n), d_(d), d2_(d2), item_set_(item_set), i2v_set_(i2v_set),
    leaf_(LEAF_SIZE), index_(nullptr)
{
    srand(RANDOM_SEED);             // setup random seed
    
    index_ = new int[n];
    int i = 0; 
    std::iota(index_, index_+n, i++);
    root_ = build(n, index_, item_set);
}

// -----------------------------------------------------------------------------
BC_Node* BC_Tree::build(            // build a bc-node 
    int   n,                            // item cardinality
    int   *index,                       // item index (allow modify)
    const float *item_set)              // item set
{
    BC_Node *cur = nullptr;
    if (n <= leaf_) {
        // build leaf node
        cur = new BC_Node(n, d_, d2_, true, nullptr, nullptr, index, item_set, i2v_set_);
    }
    else {
        float *w = new float[d_];
        int left = 0, right = n-1, cnt = 0;
        do {
            // build internal node
            int x_p = rand() % n;
            int l_p = find_furthest_id(x_p, n, index, item_set);
            int r_p = find_furthest_id(l_p, n, index, item_set);
            assert(l_p != r_p);
            
            // note: we use l_p and r_p as two pivots
            const float *l_pivot = item_set + (u64) index[l_p]*d_;
            const float *r_pivot = item_set + (u64) index[r_p]*d_;
            float l_sqr = 0.0f, r_sqr = 0.0f;
            for (int i = 0; i < d_; ++i) {
                float l_v = l_pivot[i], r_v = r_pivot[i];
                w[i] = r_v - l_v; l_sqr += SQR(l_v); r_sqr += SQR(r_v);
            }
            float b = 0.5f * (l_sqr - r_sqr);
    
            left = 0, right = n-1;
            while (left <= right) {
                const float *x = item_set + (u64) index[left]*d_;
                float val = calc_inner_product(d_, w, x) + b;
                if (val < 0.0f) ++left;
                else { SWAP(index[left], index[right]); --right; }
            }
            // if (left <= 0 || left >= n) {
            //     printf("n=%d, left=%d, right=%d\n", n, left, right);
            // }
            ++cnt;
        } while ((left <= 0 || left >= n) && cnt <= 3);
        if (cnt > 3) left = n/2; // ensure split into two parts
        delete[] w;

        BC_Node *lc = build(left, index, item_set);
        BC_Node *rc = build(n-left, index+left, item_set);
        cur = new BC_Node(n, d_, d2_, false, lc, rc, index, item_set, i2v_set_);
    }
    return cur;
}

// -----------------------------------------------------------------------------
int BC_Tree::find_furthest_id(      // find furthest item id
    int   from,                         // input item id
    int   n,                            // number of item index
    int   *index,                       // item index
    const float *item_set)              // item set
{
    int   far_id   = -1;
    float far_dist = -1.0f;

    const float *query = item_set + (u64) index[from]*d_;
    for (int i = 0; i < n; ++i) {
        if (i == from) continue;
        
        const float *item = item_set + (u64) index[i]*d_;
        float dist = calc_l2_sqr(d_, item, query);
        if (far_dist < dist) { far_dist = dist; far_id = i; }
    }
    return far_id;
}

// -----------------------------------------------------------------------------
BC_Tree::~BC_Tree()                 // destructor
{
    if (root_  != nullptr) { delete   root_;  root_  = nullptr; }
    if (index_ != nullptr) { delete[] index_; index_ = nullptr; }
}

// -----------------------------------------------------------------------------
void BC_Tree::display()             // display bc-tree
{
    std::vector<int> leaf_size;
    root_->traversal(leaf_size);
        
    printf("Parameters of BC_Tree:\n");
    printf("n        = %d\n", n_);
    printf("d        = %d\n", d_);
    printf("leaf     = %d\n", leaf_);
    printf("# leaves = %d\n", (int) leaf_size.size());
    // for (int leaf : leaf_size) printf("%d\n", leaf);
    printf("\n");
    std::vector<int>().swap(leaf_size);
}

// -----------------------------------------------------------------------------
int BC_Tree::mips(                  // mips on bc-tree (for avg & max)
    float norm_q,                       // l2-norm of query
    const float *query,                 // query (user) vector
    int   &max_id,                      // max item id (return)
    float &tau,                         // max ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip)                          // ip between item and query (return)
{
    float cq_ip = calc_inner_product(d_, root_->center_, query);
    root_->mips(cq_ip, norm_q, query, max_id, tau, check, ip);
    
    return 0;
}

// -----------------------------------------------------------------------------
int BC_Tree::mips(                  // mips on bc-tree (for avg & max)
    float norm_q,                       // l2-norm of query
    const float *query,                 // query (user) vector
    MaxK_List *list,                    // top-k results (return)
    bool  *check,                       // ip is checked? (return)
    float *ip)                          // ip between item and query (return)
{
    float cq_ip = calc_inner_product(d_, root_->center_, query);
    root_->mips(cq_ip, norm_q, query, list, check, ip);
    
    return 0;
}

// -----------------------------------------------------------------------------
int BC_Tree::diverse_mips(          // diverse mips on bc-tree (for avg)
    int   num,                          // num of id in result set
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_sum,                   // local max sum of ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *sum_ip)                     // sum of ip (return)
{
    float cq_ip = calc_inner_product(d_, root_->center_, query);
    root_->diverse_mips(num, cq_ip, f1, f2, norm_q, added, query, items, 
        max_id, tau, local_sum, check, ip, sum_ip);
    
    return 0;
}

// -----------------------------------------------------------------------------
int BC_Tree::diverse_mips(          // diverse mips on bc-tree (for max)
    int   num,                          // num of id in result set
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float max_pair_ip,                  // max pair-wise ip
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_max,                   // local max ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *max_ip)                     // max ip (return)
{
    float cq_ip = calc_inner_product(d_, root_->center_, query);
    root_->diverse_mips(num, cq_ip, f1, f2, max_pair_ip, norm_q, added, query, 
        items, max_id, tau, local_max, check, ip, max_ip);
    
    return 0;
}

// -----------------------------------------------------------------------------
//  the following two methods consider two spaces
// -----------------------------------------------------------------------------
int BC_Tree::diverse_mips_i2v(      // diverse mips on bc-tree (for avg)
    int   num,                          // num of id in result set
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_sum,                   // local max sum of ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *sum_ip)                     // sum of ip (return)
{
    float cq_ip = calc_inner_product(d_, root_->center_, query);
    root_->diverse_mips_i2v(num, cq_ip, f1, f2, norm_q, added, query, items, 
        max_id, tau, local_sum, check, ip, sum_ip);
    
    return 0;
}

// -----------------------------------------------------------------------------
int BC_Tree::diverse_mips_i2v(      // diverse mips on bc-tree (for max)
    int   num,                          // num of id in result set
    float f1,                           // balance factor for 1st term
    float f2,                           // scale   factor for 2nd term
    float max_pair_ip,                  // max pair-wise ip
    float norm_q,                       // l2-norm of query
    const bool  *added,                 // added item
    const float *query,                 // query (user) vector
    const float *items,                 // result items
    int   &max_id,                      // max item id (return)
    float &tau,                         // max score (return)
    float &local_max,                   // local max ip (return)
    bool  *check,                       // ip is checked? (return)
    float *ip,                          // ip between item and query (return)
    Result *max_ip)                     // max ip (return)
{
    float cq_ip = calc_inner_product(d_, root_->center_, query);
    root_->diverse_mips_i2v(num, cq_ip, f1, f2, max_pair_ip, norm_q, added, query, 
        items, max_id, tau, local_max, check, ip, max_ip);
    
    return 0;
}

// -----------------------------------------------------------------------------
void BC_Tree::traversal(            // traversal bc-tree to get leaf info
    std::vector<int> &leaf_size,        // leaf size (return)
    std::vector<int> &index)            // item index with leaf order (return)
{
    for (int i = 0; i < n_; ++i) index[i] = index_[i];
    root_->traversal(leaf_size);
}

} // end namespace ip