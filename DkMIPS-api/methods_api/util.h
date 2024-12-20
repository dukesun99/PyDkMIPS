#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <time.h>
#include <random>
#include <unordered_map>
#include <set>
#include <chrono>
#include <numeric>

#include <omp.h>
#include <inttypes.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

#include "def.h"
#include "pri_queue.h"

namespace ip {

// -----------------------------------------------------------------------------
//  Input & Output
// -----------------------------------------------------------------------------
void create_dir(                    // create dir if the path exists
    char *path);                        // input path

// -----------------------------------------------------------------------------
int read_bin_data(                  // read binary data from disk
    int   n,                            // number of data
    int   d,                            // dimensionality
    const char *fname,                  // address of data
    float *data);                       // data (return)

// -----------------------------------------------------------------------------
void get_csv_from_line(             // get an array with csv format from a line
    std::string str_data,               // a string line
    std::vector<int> &csv_data);        // csv data (return)

// -----------------------------------------------------------------------------
int get_conf(                       // get cand list from configuration file
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    std::vector<int> &cand);            // candidates list (return)

// -----------------------------------------------------------------------------
void write_index_info(              // display & write indexing overhead
    float index_time,                   // index time (ms)
    u64   index_size,                   // index size (bytes)
    FILE  *fp);                         // file pointer (return)

// -----------------------------------------------------------------------------
void display_head(                  // display head
    int  k,                             // top-k value
    FILE *fp);                          // file pointer

// -----------------------------------------------------------------------------
void display_head(                  // display head
    int  k,                             // top-k value
    int  cand,                          // number of candidates
    FILE *fp);                          // file pointer

// -----------------------------------------------------------------------------
void display_results(               // display results for each user (query)
    FILE  *fp,                          // file pointer
    int   n,                            // number of items
    int   k,                            // result set size
    int   uid,                          // user id
    float mmr,                          // mmr
    float relevance,                    // relevant value
    float diversity,                    // diverse value
    float msec,                         // milliseconds (ms)
    const Result *res);                 // result set

// -----------------------------------------------------------------------------
//  Distance and similarity functions
// -----------------------------------------------------------------------------
float calc_l2_sqr(                  // calc l_2 distance square
    int   dim,                          // dimensionality
    const float *p1,                    // 1st point
    const float *p2);                   // 2nd point

// -----------------------------------------------------------------------------
float calc_l2_dist(                 // calc l_2 distance
    int   dim,                          // dimensionality
    const float *p1,                    // 1st point
    const float *p2);                   // 2nd point

// -----------------------------------------------------------------------------
float calc_inner_product(           // calc inner product
    int   dim,                          // dimensionality
    const float *p1,                    // 1st point
    const float *p2);                   // 2nd point

// -----------------------------------------------------------------------------
float calc_cosine_angle(            // calc cosine angle, [-1,1]
    int   dim,                          // dimensionality
    const float *p1,                    // 1st point
    const float *p2);                   // 2nd point

// -----------------------------------------------------------------------------
float calc_angle(                   // calc angle between two points
    int   dim,                          // dimension
    const float *p1,                    // 1st point
    const float *p2);                   // 2nd point

// -----------------------------------------------------------------------------
float calc_p2h_dist(                // calc p2h dist
    int   dim,                          // dimension
    const float *p1,                    // 1st point
    const float *p2);                   // 2nd point

// -----------------------------------------------------------------------------
void calc_centroid(                 // calc centroid
    int   n,                            // number of data points
    int   dim,                          // dimensionality
    const float *data,                  // data points
    float *centroid);                   // centroid (return)

// -----------------------------------------------------------------------------
void calc_centroid(                 // calc centroid
    int   n,                            // number of data points
    int   d,                            // dimension
    const int   *index,                 // data index
    const float *data,                  // data points
    float *centroid);                   // centroid (return)

// -----------------------------------------------------------------------------
float shift_data_and_norms(         // calc shifted data and their l2-norm sqrs
    int   n,                            // number of data points
    int   d,                            // dimensionality
    const float *data,                  // data points
    const float *centroid,              // centroid
    float *shift_data,                  // shifted data points (return)
    float *shift_norms);                // shifted l2-norm sqrs (return)

// -----------------------------------------------------------------------------
//  Generate random variables
// -----------------------------------------------------------------------------
float uniform(                      // r.v. from Uniform(min, max)
    float min,                          // min value
    float max);                         // max value

// -----------------------------------------------------------------------------
//  Given a mean and a standard deviation, gaussian generates a normally 
//  distributed random number.
//
//  Algorithm:  Polar Method, p.104, Knuth, vol. 2
// -----------------------------------------------------------------------------
float gaussian(                     // r.v. from Gaussian(mean, sigma)
    float mean,                         // mean value
    float sigma);                       // std value

// -----------------------------------------------------------------------------
inline float normal_pdf(            // pdf of Guassian(mean, std)
    float x,                            // variable
    float u,                            // mean
    float sigma)                        // standard error
{
    float ret = exp(-(x - u) * (x - u) / (2.0f * sigma * sigma));
    ret /= sigma * sqrt(2.0f * PI);
    
    return ret;
}

// -----------------------------------------------------------------------------
float normal_cdf(                   // cdf of N(0, 1) in range (-inf, x]
    float x,                            // integral border
    float step);                        // step increment

// -----------------------------------------------------------------------------
float new_cdf(                      // cdf of N(0, 1) in range [-x, x]
    float x,                            // integral border
    float step);                        // step increment

// -----------------------------------------------------------------------------
//  other utility functions
// -----------------------------------------------------------------------------
float get_wc_time(                  // get actual wall clock time 
    std::chrono::system_clock::time_point start, // start time
    std::chrono::system_clock::time_point end);  // end time

} // end namespace ip
