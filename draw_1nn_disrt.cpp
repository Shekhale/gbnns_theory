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
    bool synthetic_data = true;
    if (argc == 3) {
        d_v = atoi(argv[1]);
        synthetic_data = atoi(argv[2]);
    } else {
        cout << " Need to specify parameters" << endl;
        return 1;
    }

    time_t start, end;
    const size_t n = 1000000;  // number of points in base set
    const size_t n_q = 10000;  // number of points in query set
    const size_t n_tr = 1;
    const size_t d = d_v;  // dimension of data

    L2Metric l2 = L2Metric();

    std::mt19937 random_gen;
    std::random_device device;
    random_gen.seed(device());

    string path_data = "data/synthetic/synthetic";
    string path_models = "models/synthetic/";
    const char *database_dir;
    string dir_d;
    if (synthetic_data) {
        dir_d = path_data + "_database_n_10_6_d_" + to_string(d) + ".fvecs";
        database_dir = dir_d.c_str();  // path to data
    } else {
        dir_d = "data/sift/sift_base.fvecs";
        database_dir = dir_d.c_str();  // path to data

    }
    string dir_q = path_data + "_query_n_10_4_d_" + to_string(d) + ".fvecs";
    const char *query_dir = dir_q.c_str();  // path to data
    string dir_t = path_data + "_groundtruth_n_10_4_d_" + to_string(d) + ".ivecs";
    const char *truth_dir = dir_t.c_str();  // path to data

    string output = "results/distr_to_1_d_" + to_string(d) + ".txt";
    const char *output_txt = output.c_str();

    remove(output_txt);

    bool data_exist = FileExist(dir_d);
    if (data_exist != true and synthetic_data) {
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
        vector<uint32_t> truth = get_truth(db, queries, n, d, n_q, &l2);

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

    FindDistanceToKNeighbor(n, d, 1, 10000, db, output_txt, &l2);

    return 0;

}
