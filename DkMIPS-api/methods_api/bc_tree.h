#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <numeric>
#include <cstddef>

#include "def.h"
#include "pri_queue.h"
#include "util.h"

namespace ip {

// -----------------------------------------------------------------------------
//  BC_Node: leaf node and internal node of BC_Tree
// -----------------------------------------------------------------------------
class BC_Node {
public:
    int   n_;                       // item cardinality
    int   d_;                       // dimensionality
    int   d2_;                       // dimensionality for 2nd space
    float radius_;                  // radius of a bc-node
    float *center_;                 // center of a bc-node
    
    BC_Node *lc_;                   // left  child
    BC_Node *rc_;                   // right child
    int   *index_;                  // item index (reorder for leaf only)
    float *item_set_;               // item set   (init for leaf only)
    const float *i2v_set_;          // i2v  set by item2vec
    
    float norm_c_;                  // l2-norm of center (for leaf only)
    float *r_x_;                    // radius for items  (for leaf only)
    float *x_cos_;                  // ||x|| cos(\thata) (for leaf only)
    float *x_sin_;                  // ||x|| sin(\thata) (for leaf only)
    
    // -------------------------------------------------------------------------
    BC_Node(                        // constructor
        int   n,                        // item cardinality
        int   d,                        // dimensionality
        int   d2,                       // dimensionality for 2nd space
        bool  is_leaf,                  // is leaf node
        BC_Node *lc,                    // left  child
        BC_Node *rc,                    // right child
        int   *index,                   // item  index
        const float *item_set,          // item  set
        const float *i2v_set);          // i2v  set by item2vec

    // -------------------------------------------------------------------------
    ~BC_Node();                     // destructor
    
    // -------------------------------------------------------------------------
    void mips(                      // mips on bc-node (for avg & max)
        float cq_ip,                    // inner product of center & query
        float norm_q,                   // l2-norm of query
        const float *query,             // query (user) vector
        int   &max_id,                  // max item id (return)
        float &tau,                     // max ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip);                     // ip between item and query (return)

    void linear_scan(               // linear scan (for avg & max)
        float cq_ip,                    // inner product of center & query
        float norm_q,                   // l2-norm of query
        const float *query,             // query (user) vector
        int   &max_id,                  // max item id (return)
        float &tau,                     // max ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip);                     // ip between item and query (return)
    
    // -------------------------------------------------------------------------
    void mips(                      // mips on bc-node (for avg & max)
        float cq_ip,                    // inner product of center & query
        float norm_q,                   // l2-norm of query
        const float *query,             // query (user) vector
        MaxK_List *list,                // top-k results (return)
        bool  *check,                   // ip is checked? (return)
        float *ip);                     // ip between item and query (return)

    void linear_scan(               // linear scan (for avg & max)
        float cq_ip,                    // inner product of center & query
        float norm_q,                   // l2-norm of query
        const float *query,             // query (user) vector
        MaxK_List *list,                // top-k results (return)
        bool  *check,                   // ip is checked? (return)
        float *ip);                     // ip between item and query (return)
    
    // -------------------------------------------------------------------------
    void diverse_mips(              // diverse mips on bc-node (for avg)
        int   num,                      // num of id in result set
        float cq_ip,                    // inner product of center & query
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item 
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_sum,               // local max sum of ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *sum_ip);                // sum of ip (return)
    
    void diverse_linear_scan(       // diverse linear scan (for avg)
        int   num,                      // num of id in result set
        float cq_ip,                    // inner product of center & query
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item 
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_sum,               // local max sum of ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *sum_ip);                // sum of ip (return)
    
    // -------------------------------------------------------------------------
    void diverse_mips(              // diverse mips on bc-node (for max)
        int   num,                      // num of id in result set
        float cq_ip,                    // inner product of center & query
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float max_pair_ip,              // max pair-wise ip
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_max,               // local max ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *max_ip);                // max ip (return)
    
    void diverse_linear_scan(       // diverse linear scan (for max)
        int   num,                      // num of id in result set
        float cq_ip,                    // inner product of center & query
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float max_pair_ip,              // max pair-wise ip
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_max,               // local max ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *max_ip);                // max ip (return)
    
    // -------------------------------------------------------------------------
    //  the following two methods consider two spaces
    // -------------------------------------------------------------------------
    void diverse_mips_i2v(          // diverse mips on bc-node (for avg)
        int   num,                      // num of id in result set
        float cq_ip,                    // inner product of center & query
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item 
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_sum,               // local max sum of ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *sum_ip);                // sum of ip (return)
    
    void diverse_linear_scan_i2v(   // diverse linear scan (for avg)
        int   num,                      // num of id in result set
        float cq_ip,                    // inner product of center & query
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item 
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_sum,               // local max sum of ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *sum_ip);                // sum of ip (return)
    
    // -------------------------------------------------------------------------
    void diverse_mips_i2v(          // diverse mips on bc-node (for max)
        int   num,                      // num of id in result set
        float cq_ip,                    // inner product of center & query
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float max_pair_ip,              // max pair-wise ip
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_max,               // local max ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *max_ip);                // max ip (return)
    
    void diverse_linear_scan_i2v(   // diverse linear scan (for max)
        int   num,                      // num of id in result set
        float cq_ip,                    // inner product of center & query
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float max_pair_ip,              // max pair-wise ip
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_max,               // local max ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *max_ip);                // max ip (return)
    
    // -------------------------------------------------------------------------
    float calc_sum_ip(              // calc sum of ip for input id (for avg)
        int   num,                      // num of id in result set
        int   id,                       // input id
        int   d,                        // item dimension
        const float *target,            // target item by input id
        const float *items,             // result items
        Result *sum_ip);                // sum of ip (return)
    
    float find_max_ip(              // find max ip for input id (for max)
        int   num,                      // num of id in result set
        int   id,                       // input id
        int   d,                        // item dimension
        const float *target,            // target item by input id
        const float *items,             // result items
        Result *max_ip);                // max ip (return)
    
    void traversal(                 // traversal bc-tree
        std::vector<int> &leaf_size);   // leaf size (return)
    
    // -------------------------------------------------------------------------
    inline u64 get_estimated_memory() { // get memory usage
        u64 ret = sizeof(*this) + sizeof(float)*d_; // center_
        
        // x_cos_, x_sin_, r_x_
        if (item_set_ != nullptr) return ret+sizeof(float)*n_*3;
        else return ret+lc_->get_estimated_memory()+rc_->get_estimated_memory();
    }
};

// -----------------------------------------------------------------------------
//  BC_Tree maintains a ball structure for internal nodes and a joint structure 
//  of ball & cone for leaf nodes for k-mips
// -----------------------------------------------------------------------------
class BC_Tree {
public:
    int   n_;                       // item cardinality
    int   d_;                       // dimensionality
    int   d2_;                      // dimensionality for 2nd space
    const float *item_set_;         // item set by MF
    const float *i2v_set_;          // i2v  set by item2vec
    
    int leaf_;                      // leaf size of bc-tree
    int *index_;                    // item index (allow modify)
    BC_Node *root_;                 // root node of bc-tree
    
    // -------------------------------------------------------------------------
    BC_Tree(                        // constructor
        int   n,                        // item cardinality
        int   d,                        // dimensionality
        int   d2,                       // dimensionality for 2nd space
        const float *item_set,          // item set
        const float *i2v_set);          // i2v  set by item2vec
    
    // -------------------------------------------------------------------------
    ~BC_Tree();                     // destructor
    
    // -------------------------------------------------------------------------
    void display();                 // display bc-tree
    
    // -------------------------------------------------------------------------
    int mips(                       // mips on bc-tree (for avg & max)
        float norm_q,                   // l2-norm of query
        const float *query,             // query (user) vector
        int   &max_id,                  // max item id (return)
        float &tau,                     // max ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip);                     // ip between item and query (return)
    
    // -------------------------------------------------------------------------
    int mips(                       // mips on bc-tree (for avg & max)
        float norm_q,                   // l2-norm of query
        const float *query,             // query (user) vector
        MaxK_List *list,                // top-k results (return)
        bool  *check,                   // ip is checked? (return)
        float *ip);                     // ip between item and query (return)

    // -------------------------------------------------------------------------
    int diverse_mips(               // diverse mips on bc-tree (for avg)
        int   num,                      // num of id in result set
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_sum,               // local max sum of ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *sum_ip);                // sum of ip (return)
        
    // -------------------------------------------------------------------------
    int diverse_mips(               // diverse mips on bc-tree (for max)
        int   num,                      // num of id in result set
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float max_pair_ip,              // max pair-wise ip
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_max,               // local max ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *max_ip);                // max ip (return)
    
    // -------------------------------------------------------------------------
    //  the following two methods consider two spaces
    // -------------------------------------------------------------------------
    int diverse_mips_i2v(           // diverse mips on bc-tree (for avg)
        int   num,                      // num of id in result set
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_sum,               // local max sum of ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *sum_ip);                // sum of ip (return)
        
    // -------------------------------------------------------------------------
    int diverse_mips_i2v(           // diverse mips on bc-tree (for max)
        int   num,                      // num of id in result set
        float f1,                       // balance factor for 1st term
        float f2,                       // scale   factor for 2nd term
        float max_pair_ip,              // max pair-wise ip
        float norm_q,                   // l2-norm of query
        const bool  *added,             // added item
        const float *query,             // query (user) vector
        const float *items,             // result items
        int   &max_id,                  // max item id (return)
        float &tau,                     // max score (return)
        float &local_max,               // local max ip (return)
        bool  *check,                   // ip is checked? (return)
        float *ip,                      // ip between item and query (return)
        Result *max_ip);                // max ip (return)
    
    // -------------------------------------------------------------------------
    inline u64 get_estimated_memory() { // get memory usage
        u64 ret = sizeof(*this);
        ret += root_->get_estimated_memory(); 
        return ret;
    }
    
protected:
    // -------------------------------------------------------------------------
    BC_Node* build(                 // build a bc-node
        int   n,                        // item cardinality
        int   *index,                   // item index  (allow modify)
        const float *item_set);         // item set

    // -------------------------------------------------------------------------
    int find_furthest_id(           // find furthest item id
        int   from,                     // input item id
        int   n,                        // number of item index
        int   *index,                   // item index
        const float *item_set);         // item set
    
    // -------------------------------------------------------------------------
    void traversal(                 // traversal bc-tree to get leaf info
        std::vector<int> &leaf_size,    // leaf size (return)
        std::vector<int> &index);       // item index with leaf order (return)
};

} // end namespace ip