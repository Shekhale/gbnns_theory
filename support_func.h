#include <random>
#include <iostream>
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
#include <cassert>

#include <limits>
#include <sys/time.h>

using namespace std;

float EPS = 1e-10;

struct neighbor {
    int number;
    float dist;

    size_t operator()(const neighbor &n) const {
        size_t x = std::hash<int>()(n.number);

        return x;
    }
};

bool operator<(const neighbor& x, const neighbor& y)
{
    return x.dist < y.dist;
}

class Metric {
public:
    virtual float Dist(const float *x, const float *y, size_t d) = 0;
};


class LikeL2Metric : public Metric {
public:
    float Dist(const float *x, const float *y, size_t d) {
        float res = 0;
        for (int i = 0; i < d; ++i) {
            res += pow(*x - *y, 2);
            ++x;
            ++y;
        }
        return res;
    }
};


template<typename T>
void readXvec(std::ifstream &in, T *data, const size_t d, const size_t n = 1)
{
    uint32_t dim = d;
    for (size_t i = 0; i < n; i++) {
        in.read((char *) &dim, sizeof(uint32_t));
        if (dim != d) {
            std::cout << "file error\n";
            std::cout << "dim " << dim << ", d " << d << std::endl;
            std::cout << "our fault\n";

            exit(1);
        }
        in.read((char *) (data + i * dim), dim * sizeof(T));
    }
}


template<typename T>
void writeXvec(std::ofstream &out, T *data, const size_t d, const size_t n = 1)
{
    const uint32_t dim = d;
    for (size_t i = 0; i < n; i++) {
        out.write((char *) &dim, sizeof(uint32_t));
        out.write((char *) (data + i * dim), dim * sizeof(T));
    }
}

void write_edges(const char *location, const std::vector<std::vector<uint32_t>> &edges) {
    std::cout << "Saving edges to " << location << std::endl;
    std::ofstream output(location, std::ios::binary);

    for (uint32_t i = 0; i < edges.size(); i++) {
        const uint32_t *data = edges[i].data();
        uint32_t size = edges[i].size();

        output.write((char *) &size, sizeof(uint32_t));
        output.write((char *) data, sizeof(uint32_t) * size);
    }
}


vector<std::vector<uint32_t>> load_edges(const char *location, std::vector<std::vector<uint32_t>> edges) {
    // std::cout << "Loading edges from " << location << std::endl;
    std::ifstream input(location, std::ios::binary);

    uint32_t size;
    for (int i = 0; i < edges.size(); i++) {
        input.read((char *) &size, sizeof(uint32_t));

        vector<uint32_t> vec(size);
        uint32_t *data = vec.data();
        input.read((char *) data, sizeof(uint32_t)*size);
        for (int j = 0; j < size; ++j) {
            edges[i].push_back(vec[j]);
        }
    }
    return edges;
}


vector<float> create_uniform_data(int N, int d, std::mt19937 random_gen) {
    vector<float> ds(N*d);
    normal_distribution<float> norm_distr(0, 1);
    for (int i=0; i < N; ++i) {
        vector<float> point(d);
        float norm_coeff = 0;
        for (int j=0; j < d; ++j) {
            point[j] = norm_distr(random_gen);
            norm_coeff += point[j] * point[j];
        }
        norm_coeff = pow(norm_coeff, 0.5);
        for (int j=0; j < d; ++j) {
            ds[i * d + j] = point[j] / norm_coeff;
        }
    }
    return  ds;
}

vector<uint32_t> get_truth(vector<float> ds, vector<float> query, int N, int d, int N_q, Metric *metric) {
    vector<uint32_t> truth(N_q);
#pragma omp parallel for
    for (int i=0; i < N_q; ++i) {
        const float* tendered_d = ds.data();
        const float* goal = query.data() + i*d;
        float dist = metric->Dist(tendered_d, goal, d);
        float new_dist = dist;
        int tendered_num = 0;
        for (int j=1; j<N; ++j) {
            tendered_d += d;
            new_dist = metric->Dist(tendered_d, goal, d);
            if (new_dist < dist) {
                dist = new_dist;
                tendered_num = j;
            }
        }
        truth[i] = tendered_num;
    }
    return truth;
}

vector< vector <uint32_t>> CutKNNbyThreshold(vector< vector <uint32_t>> &knn, vector<float> &ds, float thr, int N, int d,
                                  Metric *metric) {
    vector< vector <uint32_t>> knn_cut(N);
#pragma omp parallel for
    for (int i=0; i < N; ++i) {
        const float* point_i = ds.data() + i*d;
        for (int j=0; j < knn[i].size(); ++j) {
            int cur = knn[i][j];
            const float *point_cur = ds.data() + cur*d;
            if (metric->Dist(point_i, point_cur, d) < thr) {
                knn_cut[i].push_back(cur);
            }
        }
    }
    return  knn_cut;
}

vector< vector <uint32_t>> CutKNNbyK(vector< vector <uint32_t>> &knn, vector<float> &ds, int knn_size, int N, int d,
                                             Metric *metric) {
    vector< vector <uint32_t>> knn_cut(N);
    bool small_size = false;
    #pragma omp parallel for
    for (int i=0; i < N; ++i) {
        vector<neighbor> neigs;
        const float* point_i = ds.data() + i*d;
        for (int j=0; j < knn[i].size(); ++j) {
            int cur = knn[i][j];
            const float *point_cur = ds.data() + cur*d;
            neighbor neig{cur, metric->Dist(point_i, point_cur, d)};
            neigs.push_back(neig);
        }
        if (not small_size and knn_size > knn[i].size()) {
            cout << "Size knn less than you want" << endl;
            cout << knn[i].size() << endl;
            //exit(1);
            small_size = true;
        }

        sort(neigs.begin(), neigs.end());
        int cur_size = knn_size;
        if (knn[i].size() < cur_size) {
			cur_size = knn[i].size();
		}
        for (int j=0; j < cur_size; ++j) {
            knn_cut[i].push_back(neigs[j].number);
        }
    }
    return  knn_cut;
}


vector< vector <uint32_t>> CutKL(vector< vector <uint32_t>> &kl, int l, int N, vector< vector <uint32_t>> &knn) {
    vector< vector <uint32_t>> kl_cut(N);
    #pragma omp parallel for
    for (int i=0; i < N; ++i) {
        if (l > kl[i].size()) {
            cout << "Graph have less edges that you want" << endl;
            exit(1);
        }
        vector <uint32_t> kl_sh = kl[i];
        random_shuffle(kl_sh.begin(), kl_sh.end());
        int it = 0;
        while (kl_cut[i].size() < l and it < kl_sh.size()) {
            if (find(knn[i].begin(), knn[i].end(), kl_sh[it]) == knn[i].end()) {
                kl_cut[i].push_back(kl_sh[it]);
            }
            ++it;
        }
    }
    return  kl_cut;
}


int FindGraphAverageDegree(vector< vector <uint32_t>> &graph) {
    float ans = 0;
    int n = graph.size();
    for (int i=0; i < n; ++i) {
        ans += graph[i].size();
    }
    return float(ans / n);
}


inline bool FileExist (std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

vector< vector<uint32_t> > GraphMerge(vector< vector<uint32_t> > &graph_f, vector< vector<uint32_t> > &graph_s) {
    int n = graph_f.size();
    vector <vector<uint32_t> > union_graph(n);
#pragma omp parallel for
    for (int i=0; i < n; ++i) {
        for (int j =0; j < graph_f[i].size(); ++j) {
            union_graph[i].push_back(graph_f[i][j]);
        }
        for (int j =0; j < graph_s[i].size(); ++j) {
            if (find(union_graph[i].begin(), union_graph[i].end(), graph_s[i][j]) == union_graph[i].end()) {
                union_graph[i].push_back(graph_s[i][j]);
            }
        }
    }

    return union_graph;
}
