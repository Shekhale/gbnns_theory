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
    size_t knn_size_for_beam = 20;
    if (d == 3) {
        knn_size = 20;
        knn_size_for_beam = 8;
    } else if (d == 5) {
        knn_size = 60;
        knn_size_for_beam = 10;
    } else if (d == 9) {
        knn_size = 300;
        knn_size_for_beam = 16;
    } else if (d == 17) {
        knn_size = 2000;
    }

    LikeL2Metric ll2 = LikeL2Metric();

    cout << "d = " << d << ", kl_size = " << kl_size << ", knn_size = " << knn_size <<  endl;

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


    string dir_kl = path_models + "kl_sqrt_style_n_10_6_d_" + to_string(d) + ".ivecs";
    const char *edge_kl_dir = dir_kl.c_str();


    string output = "results/synthetic_n_10_6_d_" + to_string(d) + ".txt";
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
        vector<uint32_t> truth = get_truth(db, queries, n, d, n_q, &ll2);

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

    std::vector<float> queries(n_q * d);
    {
        std::ifstream data_input(query_dir, std::ios::binary);
        readXvec<float>(data_input, queries.data(), d, n_q);
    }

    std::vector<uint32_t> truth(n_q * n_tr);
    {
        std::ifstream data_input(truth_dir, std::ios::binary);
        readXvec<uint32_t>(data_input, truth.data(), n_tr, n_q);
    }


//--------------------------------------------------------------------------------------------------------------------------------------------

// BUILD GRAPHS


    bool knn_exist = FileExist(dir_knn);
    if (knn_exist != true) {
        time(&start);
        ExactKNN knn;
        knn.Build(knn_size, db, n, d, &ll2);
        cout << knn.matrixNN[0].size() << ' ' << knn.matrixNN[3].size() << endl;
        time(&end);
        cout << difftime(end, start) << endl;
        write_edges(edge_knn_dir, knn.matrixNN);
    }

    vector< vector <uint32_t>> knn(n);
    knn = load_edges(edge_knn_dir, knn);
    cout << "knn " << FindGraphAverageDegree(knn) << endl;
    knn = CutKNNbyK(knn, db, knn_size, n, d, &ll2);
    cout << "knn " << FindGraphAverageDegree(knn) << endl;

    vector< vector <uint32_t>> knn_for_beam = CutKNNbyK(knn, db, knn_size_for_beam, n, d, &ll2);
    cout << "knn_for_beam " << FindGraphAverageDegree(knn_for_beam) << endl;

    bool kl_exist = FileExist(dir_kl);
    if (kl_exist != true) {
        time(&start);
		KLgraph kl_sqrt;
		kl_sqrt.BuildByNumberCustom(kl_size, db, n, d, pow(n, 0.5), random_gen, &ll2);
		time (&end);
		cout << difftime(end, start) << endl;
        write_edges(edge_kl_dir, kl_sqrt.longmatrixNN);
    }

    vector< vector <uint32_t>> kl(n);
    kl = load_edges(edge_kl_dir, kl);
    cout << "kl " << FindGraphAverageDegree(kl) << endl;


	get_synthetic_tests(n, d, n_q, n_tr, random_gen, knn, knn, db, queries, truth, output_txt, &ll2, "knn", false, false, false, false);
	get_synthetic_tests(n, d, n_q, n_tr, random_gen, knn, kl, db, queries, truth, output_txt, &ll2, "knn_kl", true, false, false, false);
	get_synthetic_tests(n, d, n_q, n_tr, random_gen, knn, kl, db, queries, truth, output_txt, &ll2, "knn_kl_llf", true, true, false, false);
	get_synthetic_tests(n, d, n_q, n_tr, random_gen, knn_for_beam, knn_for_beam, db, queries, truth, output_txt, &ll2, "knn_beam", false, false, true, false);
	get_synthetic_tests(n, d, n_q, n_tr, random_gen, knn_for_beam, kl, db, queries, truth, output_txt, &ll2, "knn_beam_kl_llf", true, true, true, false);



    return 0;

}
