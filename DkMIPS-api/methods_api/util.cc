#include "util.h"

namespace ip {

// -----------------------------------------------------------------------------
//  Input & Output
// -----------------------------------------------------------------------------
void create_dir(                    // create dir if the path exists
    char *path)                         // input path
{
    int len = (int) strlen(path);
    for (int i = 0; i < len; ++i) {
        if (path[i] != '/') continue; 
        
        char ch = path[i+1]; path[i+1] = '\0';
        if (access(path, F_OK) != 0) { // create directory if not exists
            if (mkdir(path, 0755) != 0) {
                printf("Could not create directory %s\n", path); exit(1);
            }
        }
        path[i+1] = ch;
    }
}

// -----------------------------------------------------------------------------
int read_bin_data(                  // read binary data from disk
    int   n,                            // number of data
    int   d,                            // dimensionality
    const char *fname,                  // address of data
    float *data)                        // data (return)
{
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }
    
    fread(data, sizeof(float), (u64) n*d, fp);
    fclose(fp);
    
    // for (int i = 0; i < (u64) n*d; ++i) data[i] *= 10000.0f;
    return 0;
}

// -----------------------------------------------------------------------------
void get_csv_from_line(             // get an array with csv format from a line
    std::string str_data,               // a string line
    std::vector<int> &csv_data)         // csv data (return)
{
    csv_data.clear();

    std::istringstream ss(str_data);
    while (ss) {
        std::string s;
        if (!getline(ss, s, ',')) break;
        csv_data.push_back(stoi(s));
    }
}

// -----------------------------------------------------------------------------
int get_conf(                       // get cand list from configuration file
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    std::vector<int> &cand)             // candidates list (return)
{
    std::ifstream infile(conf_name);
    if (!infile) { printf("Could not open %s\n", conf_name); return 1; }

    std::string dname, mname, tmp;
    bool stop = false;
    while (infile) {
        getline(infile, dname);
        
        // // skip the first three methods
        // getline(infile, mname); getline(infile, tmp); getline(infile, tmp);
        // getline(infile, mname); getline(infile, tmp); getline(infile, tmp);
        // getline(infile, mname); getline(infile, tmp); getline(infile, tmp);

        // check the remaining methods
        while (true) {
            getline(infile, mname);
            if (mname.length() == 0) break;

            if ((dname.compare(data_name)==0) && (mname.compare(method_name)==0)) {
                getline(infile, tmp); get_csv_from_line(tmp, cand);
                stop = true; break;
            } else {
                getline(infile, tmp);
            }
        }
        if (stop) break;
    }
    infile.close();
    return 0;
}

// -----------------------------------------------------------------------------
void write_index_info(              // display & write indexing overhead
    float index_time,                   // index time (ms)
    u64   index_size,                   // index size (bytes)
    FILE  *fp)                          // file pointer (return)
{
    double mb_size = index_size / 1048576.0; // convert bytes into megabytes
    
    printf("\nIndexing Time: %g MSec\n", index_time);
    printf("Estimated Mem: %g MB\n\n", mb_size);

    fprintf(fp, "Indexing Time: %g MSec\n", index_time);
    fprintf(fp, "Estimated Memory: %g MB\n\n", mb_size);
}

// -----------------------------------------------------------------------------
void display_head(                  // display head
    int  k,                             // top-k value
    FILE *fp)                           // file pointer
{
    printf("-----------------------------------------------------------------"
        "-----------\n");
    printf("UID\t\tMMR\t\tRelevance\tDiversity\tMSec\n");
    printf("-----------------------------------------------------------------"
        "-----------\n");
    
    fprintf(fp, "UID,MMR,Relevance,Diversity,MSec");
    for (int i = 0; i < k; ++i) fprintf(fp, ",ID%d", i+1);
    for (int i = 0; i < k; ++i) fprintf(fp, ",IP%d", i+1);
    fprintf(fp, "\n");
}

// -----------------------------------------------------------------------------
void display_head(                  // display head
    int  k,                             // top-k value
    int  cand,                          // number of candidates
    FILE *fp)                           // file pointer
{
    printf("cand = %d\n", cand);
    printf("-----------------------------------------------------------------"
        "-----------\n");
    printf("UID\t\tMMR\t\tRelevance\tDiversity\tMSec\n");
    printf("-----------------------------------------------------------------"
        "-----------\n");
    
    fprintf(fp, "cand = %d\n", cand);
    fprintf(fp, "UID,MMR,Relevance,Diversity,MSec");
    for (int i = 0; i < k; ++i) fprintf(fp, ",ID%d", i+1);
    for (int i = 0; i < k; ++i) fprintf(fp, ",IP%d", i+1);
    fprintf(fp, "\n");
}

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
    const Result *res)                  // result set
{
    if (n > 1000000) {
        printf("%d\t%.6f\t%.6f\t%.6f\t%.6f\n", uid, mmr, relevance, 
            diversity, msec);
    } else {
        printf("%d\t\t%.6f\t%.6f\t%.6f\t%.6f\n", uid, mmr, relevance, 
            diversity, msec);
    }
    
    fprintf(fp, "%d,%g,%g,%g,%g", uid, mmr, relevance, diversity, msec);
    for (int j = 0; j < k; j++) fprintf(fp, ",%d", res[j].id_);
    for (int j = 0; j < k; j++) fprintf(fp, ",%g", res[j].key_);
    fprintf(fp, "\n");
}

// -----------------------------------------------------------------------------
//  Distance and similarity functions
// -----------------------------------------------------------------------------
float calc_l2_sqr(                  // calc l_2 distance square
    int   dim,                          // dimensionality
    const float *p1,                    // 1st point
    const float *p2)                    // 2nd point
{
    float ret = 0.0f;
    for (int i = 0; i < dim; ++i) ret += SQR(p1[i] - p2[i]);

    return ret;
}

// -----------------------------------------------------------------------------
float calc_l2_dist(                 // calc l_2 distance
    int   dim,                          // dimensionality
    const float *p1,                    // 1st point
    const float *p2)                    // 2nd point
{
    return sqrt(calc_l2_sqr(dim, p1, p2));
}

// -----------------------------------------------------------------------------
float calc_inner_product(           // calc inner product
    int   dim,                          // dimensionality
    const float *p1,                    // 1st point
    const float *p2)                    // 2nd point
{
    float ret = 0.0f;
    for (int i = 0; i < dim; ++i) ret += p1[i] * p2[i];
    
    return ret;
}

// -----------------------------------------------------------------------------
float calc_cosine_angle(            // calc cosine angle, [-1,1]
    int   dim,                          // dimensionality
    const float *p1,                    // 1st point
    const float *p2)                    // 2nd point
{
    float ip    = calc_inner_product(dim, p1, p2);
    float norm1 = calc_inner_product(dim, p1, p1);
    float norm2 = calc_inner_product(dim, p2, p2);

    return ip / sqrt(norm1 * norm2);
}

// -----------------------------------------------------------------------------
float calc_angle(                   // calc angle between two points
    int   dim,                          // dimension
    const float *p1,                    // 1st point
    const float *p2)                    // 2nd point
{
    // acos returns an angle in [0,pi]
    return acos(calc_cosine_angle(dim, p1, p2));
}

// -----------------------------------------------------------------------------
float calc_p2h_dist(                // calc p2h dist
    int   dim,                          // dimension
    const float *p1,                    // 1st point
    const float *p2)                    // 2nd point
{
    return fabs(calc_inner_product(dim, p1, p2));
}

// -----------------------------------------------------------------------------
void calc_centroid(                 // calc centroid
    int   n,                            // number of data points
    int   dim,                          // dimensionality
    const float *data,                  // data points
    float *centroid)                    // centroid (return)
{
    memset(centroid, 0.0f, sizeof(float)*dim);
    for (int i = 0; i < n; ++i) {
        const float *point = data + (u64) i*dim;
        for (int j = 0; j < dim; ++j) centroid[j] += point[j];
    }
    for (int i = 0; i < dim; ++i) centroid[i] /= (float) n;
}

// -----------------------------------------------------------------------------
void calc_centroid(                 // calc centroid
    int   n,                            // number of data points
    int   d,                            // dimension
    const int   *index,                 // data index
    const float *data,                  // data points
    float *centroid)                    // centroid (return)
{
    memset(centroid, 0.0f, sizeof(float)*d);
    for (int i = 0; i < n; ++i) {
        const float *point = data + (u64) index[i]*d;
        for (int j = 0; j < d; ++j) centroid[j] += point[j];
    }
    for (int i = 0; i < d; ++i) centroid[i] /= (float) n;
}

// -----------------------------------------------------------------------------
float shift_data_and_norms(         // calc shifted data and their l2-norm sqrs
    int   n,                            // number of data vectors
    int   d,                            // dimensionality
    const float *data,                  // data vectors
    const float *centroid,              // centroid
    float *shift_data,                  // shifted data vectors (return)
    float *shift_norms)                 // shifted l2-norm sqrs (return)
{
    float max_norm_sqr = -1.0f;
    for (int i = 0; i < n; ++i) {
        const float *record = data + (u64) i*d;
        float *shift_record = shift_data + (u64) i*d;
        
        // calc shifted data & its l2-norm
        float norm = 0.0f;
        for (int j = 0; j < d; ++j) {
            float diff = record[j] - centroid[j];
            
            shift_record[j] = diff; 
            norm += SQR(diff);
        }
        // update shift_norm & max l2-norm
        shift_norms[i] = norm;
        if (max_norm_sqr < norm) max_norm_sqr = norm;
    }
    return max_norm_sqr;
}

// -----------------------------------------------------------------------------
//  Generate random variables
// -----------------------------------------------------------------------------
float uniform(                      // r.v. from Uniform(min, max)
    float min,                          // min value
    float max)                          // max value
{
    int   num  = rand();
    float base = (float) RAND_MAX - 1.0F;
    float frac = ((float) num) / base;

    return (max - min) * frac + min;
}

// -----------------------------------------------------------------------------
//  Given a mean and a standard deviation, gaussian generates a normally 
//  distributed random number.
//
//  Algorithm:  Polar Method, p.104, Knuth, vol. 2
// -----------------------------------------------------------------------------
float gaussian(                     // r.v. from Gaussian(mean, sigma)
    float mean,                         // mean value
    float sigma)                        // std value
{
    float v1 = -1.0f, v2 = -1.0f, s = -1.0f, x = -1.0f;
    do {
        v1 = 2.0f * uniform(0.0f, 1.0f) - 1.0f;
        v2 = 2.0f * uniform(0.0f, 1.0f) - 1.0f;
        s = v1 * v1 + v2 * v2;
    } while (s >= 1.0f);
    x = v1 * sqrt(-2.0f * log(s) / s);

    // x is distributed from N(0,1)
    return x * sigma + mean;
}

// -----------------------------------------------------------------------------
float normal_cdf(                   // cdf of N(0, 1) in range (-inf, x]
    float x,                            // integral border
    float step)                         // step increment
{
    float ret = 0.0f;
    for (float i = -10.0f; i < x; i += step) {
        ret += step * normal_pdf(i, 0.0f, 1.0f);
    }
    return ret;
}

// -----------------------------------------------------------------------------
float new_cdf(                      // cdf of N(0, 1) in range [-x, x]
    float x,                            // integral border
    float step)                         // step increment
{
    float result = 0.0f;
    for (float i = -x; i <= x; i += step) {
        result += step * normal_pdf(i, 0.0f, 1.0f);
    }
    return result;
}

// -----------------------------------------------------------------------------
float get_wc_time(                  // get actual wall clock time 
    std::chrono::system_clock::time_point start, // start time
    std::chrono::system_clock::time_point end)   // end time
{
    auto  time = end - start;
    float microsec = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
    return microsec / 1000; // microsecond to millisecond
}

} // end namespace ip
