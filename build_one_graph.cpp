#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <ctime>
#include <queue>
#include <vector>
#include <omp.h>

#include <limits>
#include <sys/time.h>


#include <set>
#include <algorithm>
#include <ctime>

#include "search_function.h"

using  namespace std;

int main(int argc, char **argv) {

    size_t d_v = 3;
    if (argc == 2) {
        d_v = atoi(argv[1]);
    } else {
        cout << " Need to specify parameters" << endl;
        return 1;
    }

    time_t start, end;
    const size_t n = 1000000;  // number of points in base set
    const size_t n_q = 10000;  // number of points in query set
    const size_t n_tr = 1;
    const size_t d = d_v;  // dimension of data
    const size_t kl_size = 15; // KL graph size

    size_t knn_size = 100;
    float knn_angle = 1.25;
    if (d == 3) {
        knn_size = 20;
        knn_angle = 1.8;
    } else if (d == 5) {
        knn_size = 60;
        knn_angle = 1.8;
    } else if (d == 9) {
        knn_size = 300;
        knn_angle = 1.6;
    } else if (d == 17) {
        knn_size = 2000;
        knn_angle = 1.25;
    }
    float knn_thr = asin(knn_angle * pow(2, 0.5) * pow(n, - 1. / d));

    CosMetric cos = CosMetric();
//    LikeL2Metric ll2 = LikeL2Metric();

    cout << "d = " << d << ", kl_size = " << kl_size << ", knn_size = " << knn_size << ", knn_thr = " << knn_thr <<  endl;

    std::mt19937 random_gen;
    std::random_device device;
    random_gen.seed(device());

    string path_data = "data/synthetic/synthetic";
    string path_models = "models/synthetic/";

    string dir_d = path_data + "_database_n_10_6_d_" + to_string(d) + ".fvecs";
    const char *database_dir = dir_d.c_str();  // path to data
    string dir_q = path_data + "_query_n_10_4_d_" + to_string(d) + ".fvecs";
    const char *query_dir = dir_q.c_str();  // path to data
    string dir_t = path_data + "_groundtruth_n_10_4_d_" + to_string(d) + ".ivecs";
    const char *truth_dir = dir_t.c_str();  // path to data


    string dir_knn = path_models + "knn_n_10_6_d_" + to_string(d) + ".ivecs";
    const char *edge_knn_dir = dir_knn.c_str();
    string dir_knn_thr = path_models + "knn_thr_n_10_6_d_" + to_string(d) + "_llt.ivecs";
    const char *edge_knn_thr_dir = dir_knn_thr.c_str();


    string dir_kl_sqrt = path_models + "kl_sqrt_style_n_10_6_d_" + to_string(d) + "_llt.ivecs";
    const char *edge_kl_sqrt_dir = dir_kl_sqrt.c_str();
    string dir_kl_n = path_models + "kl_n_style_n_10_6_d_" + to_string(d) + "_llt.ivecs";
    const char *edge_kl_numb_dir = dir_kl_n.c_str();
    string dir_kl_d = path_models + "kl_dist_style_n_10_6_d_" + to_string(d) + "_llt.ivecs";
    const char *edge_kl_dist_dir = dir_kl_d.c_str();


    string output = "results/synthetic_n_10_6_d_" + to_string(d) + "_llt.txt";
    const char *output_txt = output.c_str();

    remove(output_txt);


    bool data_exist = FileExist(dir_d);
    if (data_exist != true) {
        std::cout << "Creating data" << std::endl;
        vector<float> data = create_uniform_data(n_q + n, d, random_gen);
        vector<float> queries;
        for (int i=0; i < n_q*d; ++i) {
            queries.push_back(data[i]);
        }
        vector<float> db;
        for (int i=0; i < n*d; ++i) {
            db.push_back(data[n_q*d + i]);
        }
        vector<uint32_t> truth = get_truth(db, queries, n, d, n_q, &cos);

        std::ofstream data_input_db(database_dir, std::ios::binary);
        writeXvec<float>(data_input_db, db.data(), d, n);

        std::ofstream data_input_q(query_dir, std::ios::binary);
        writeXvec<float>(data_input_q, queries.data(), d, n_q);

        std::ofstream data_input_g(truth_dir, std::ios::binary);
        writeXvec<uint32_t>(data_input_g, truth.data(), n_tr, n_q);
    }


    std::cout << "Loading data from " << database_dir << std::endl;
    std::vector<float> db(n * d);
    {
        std::ifstream data_input(database_dir, std::ios::binary);
        readXvec<float>(data_input, db.data(), d, n);
    }

    time(&start);
    KLgraph kl_n;
    kl_n.BuildByNumber(kl_size, db, n, d, random_gen, &cos);
    time (&end);
    cout << difftime(end, start) << endl;
    write_edges(edge_kl_numb_dir, kl_n.longmatrixNN);

    return 0;

}
