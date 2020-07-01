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


#include <chrono>

#include <limits>
#include <sys/time.h>


#include <algorithm>
#include <ctime>


#include "support_classes.h"
#include "visited_list_pool.h"


using namespace std;


struct triple_result {
    priority_queue<pair<float, int > > topk;
    int hops;
    int dist_calc;
    int degree;
};


void MakeStep(vector <uint32_t> &graph_level, const float *query, const float* db,
              priority_queue<pair<float, int > > &topResults,
              priority_queue<std::pair<float, int > > &candidateSet,
              Metric *metric, uint32_t d, int &query_dist_calc, bool &found, int &ef, int &k,
              VisitedList *vl) {


    vl_type *massVisited = vl->mass;
    vl_type currentV = vl->curV;
    for (int j = 0; j < graph_level.size(); ++j) {
        int neig_num = graph_level[j];
        if (massVisited[neig_num] != currentV) {
            massVisited[neig_num] = currentV;
            const float *neig_coord = db + neig_num * d;
            float dist = metric->Dist(query, neig_coord, d);
            query_dist_calc++;

            if (topResults.top().first > dist || topResults.size() < ef) {
                candidateSet.emplace(-dist, neig_num);
                found = true;
                topResults.emplace(dist, neig_num);
                if (topResults.size() > ef)
                    topResults.pop();
            }
        }
    }
}


triple_result search(const float *query, const float* db, uint32_t N, uint32_t d,
                      vector<vector <uint32_t> > &main_graph, vector<vector <uint32_t> > &auxiliary_graph,
                      int ef, int k, vector<uint32_t> &inter_points, Metric *metric,
                     VisitedListPool *visitedlistpool,
                      bool use_second_graph, bool llf, uint32_t hops_bound) {


    std::priority_queue<std::pair<float, int > > topResults;

    int query_dist_calc = 1;
    int num_hops = 0;
    for (int i = 0; i < inter_points.size(); ++i) {
        std::priority_queue<std::pair<float, int > > candidateSet;
        const float* start = db + inter_points[i]*d;
        float dist = metric->Dist(query, start, d);

        topResults.emplace(dist, inter_points[i]);
        candidateSet.emplace(-dist, inter_points[i]);
        VisitedList *vl = visitedlistpool->getFreeVisitedList();
        vl_type *massVisited = vl->mass;
        vl_type currentV = vl->curV;
        massVisited[inter_points[i]] = currentV;
        while (!candidateSet.empty()) {
            std::pair<float, int> curr_el_pair = candidateSet.top();
            if (-curr_el_pair.first > topResults.top().first) break;

            candidateSet.pop();
            int curNodeNum = curr_el_pair.second;
            bool auxiliary_found = false;

            if (use_second_graph and num_hops < hops_bound) {
                vector <uint32_t> curAuxiliaryNodeNeighbors = auxiliary_graph[curNodeNum];
                MakeStep(curAuxiliaryNodeNeighbors, query, db,
                        topResults, candidateSet,
                        metric,
                        d, query_dist_calc, auxiliary_found, ef, k,
                        vl);
            }

            if (!(auxiliary_found * llf) or !use_second_graph) {
                vector <uint32_t> curMainNodeNeighbors = main_graph[curNodeNum];
                MakeStep(curMainNodeNeighbors, query, db,
                        topResults, candidateSet,
                        metric,
                        d, query_dist_calc, auxiliary_found, ef, k,
                        vl);
            }
            num_hops++;
        }
        visitedlistpool->releaseVisitedList(vl);
    }


    while (topResults.size() > k) {
        topResults.pop();
    }

    triple_result ans{topResults, num_hops, query_dist_calc};
    return ans;
}


int GetRealNearest(const float* point_q, int k, int d, int d_low, priority_queue<pair<float, int > > &topk,
                    vector<float> &ds,
                    Metric *metric) {

    const float* point_i = ds.data() + d * topk.top().second;
    float min_dist = metric->Dist(point_i, point_q, d);
    int real_topk = topk.top().second;
    topk.pop();
    float dist;
    while (!topk.empty()) {
        point_i = ds.data() + d * topk.top().second;
        dist = metric->Dist(point_i, point_q, d);
        if (dist < min_dist) {
            min_dist = dist;
            real_topk = topk.top().second;
        }
        topk.pop();
    }

    return real_topk;
}

void get_one_test(vector<vector<uint32_t> > &knn_graph, vector<vector<uint32_t> > &kl_graph,
                  vector<float> &ds, vector<float> &queries, vector<float> &ds_low, vector<float> &queries_low,
                  vector<uint32_t> &truth,
                  int n, int d, int d_low, int n_q, int n_tr, int ef, int k, string graph_name,
                  Metric *metric, const char* output_txt,
                  vector<vector<uint32_t> > inter_points, bool use_second_graph, bool llf, uint32_t hops_bound, int dist_calc_boost,
                  int recheck_size, int number_exper, int number_of_threads) {

    std::ofstream outfile;
    outfile.open(output_txt, std::ios_base::app);


    VisitedListPool *visitedlistpool = new VisitedListPool(1, n);
    int hops = 0;
    int dist_calc = 0 + dist_calc_boost * n_q;
    float acc = 0;
    float work_time = 0;
    int num_exp = 0;

    omp_set_num_threads(number_of_threads);
    for (int v = 0; v < number_exper; ++v) {
        num_exp += 1;
        vector<int> ans(n_q);
        StopW stopw = StopW();
#pragma omp parallel for
        for (int i = 0; i < n_q; ++i) {

            triple_result tr;
            const float* point_q = queries.data() + i * d;
            const float* point_q_low = queries_low.data() + i * d_low;
            if (d != d_low) {
				if (recheck_size > 0) {
	                tr = search(point_q_low, ds_low.data(), n, d_low, knn_graph, kl_graph, recheck_size,
	                            recheck_size, inter_points[i], metric, visitedlistpool, use_second_graph, llf, hops_bound);
	                ans[i] = GetRealNearest(point_q, k, d, d_low, tr.topk, ds, metric);
	                dist_calc += recheck_size;
				} else {
					tr = search(point_q_low, ds_low.data(), n, d_low, knn_graph, kl_graph, ef,
	                            k, inter_points[i], metric, visitedlistpool, use_second_graph, llf, hops_bound);

	                while (tr.topk.size() > k) {
	                    tr.topk.pop();
	                }
	                ans[i] = tr.topk.top().second;
				}
            } else {
                tr = search(point_q, ds.data(), n, d, knn_graph, kl_graph, ef,
                            k, inter_points[i], metric, visitedlistpool, use_second_graph, llf, hops_bound);

                while (tr.topk.size() > k) {
                    tr.topk.pop();
                }
                ans[i] = tr.topk.top().second;
            }

            hops += tr.hops;
            dist_calc += tr.dist_calc;
        }

        work_time += stopw.getElapsedTimeMicro();

        int print = 0;
        for (int i = 0; i < n_q; ++i) {
            acc += ans[i] == truth[i * n_tr];
        }
    }


    cout << "graph_type " << graph_name << " acc " << acc /  (num_exp * n_q) << " hops " << hops /  (num_exp * n_q) << " dist_calc "
         << dist_calc /  (num_exp * n_q) << " work_time " << work_time / (num_exp * 1e6 * n_q) << endl;
    outfile << "graph_type " << graph_name << " acc " << acc /  (num_exp * n_q) << " hops " << hops /  (num_exp * n_q) << " dist_calc "
            << dist_calc /  (num_exp * n_q) << " work_time " << work_time / (num_exp * 1e6 * n_q) << endl;
}



void get_synthetic_tests(int n, int d, int n_q, int n_tr, std::mt19937 random_gen,
                vector< vector<uint32_t> > &knn, vector< vector<uint32_t> > &kl, vector<float> &db,
                vector<float> &queries, vector<uint32_t> &truth, const char* output_txt,
                Metric *metric, string graph_name, bool use_second_graph, bool llf, bool beam_search,
                bool knn_by_threshold) {

    vector<vector<uint32_t> > inter_points(n_q);
    int num = 0;
    uniform_int_distribution<int> uniform_distr(0, n-1);
    for (int j=0; j < n_q; ++j) {
        num = uniform_distr(random_gen);
        inter_points[j].push_back(num);
    }


    vector<int> ef_coeff;
    vector<int> k_coeff;
    vector<float> thr_coeff;
    uint32_t hops_bound = 11;
    int recheck_size = -1;
    int knn_size = FindGraphAverageDegree(knn);

    if (beam_search) {
        vector<int> k_coeff_{knn_size, knn_size, knn_size, knn_size, knn_size, knn_size};
        k_coeff.insert(k_coeff.end(), k_coeff_.begin(), k_coeff_.end());
    } else {
        vector<int> ef_coeff_{1, 1, 1, 1, 1, 1};
        ef_coeff.insert(ef_coeff.end(), ef_coeff_.begin(), ef_coeff_.end());
    }

    if (d == 3) {
        if (beam_search) {
            vector<int> ef_coeff_{10, 15, 20, 25, 30};
            ef_coeff.insert(ef_coeff.end(), ef_coeff_.begin(), ef_coeff_.end());
        } else {
            vector<int> k_coeff_{12, 14, 16, 18, 20};
            k_coeff.insert(k_coeff.end(), k_coeff_.begin(), k_coeff_.end());
            vector<float> thr_coeff_{1.1, 1.2, 1.3, 1.4, 1.5};
            thr_coeff.insert(thr_coeff.end(), thr_coeff_.begin(), thr_coeff_.end());
        }
        hops_bound = 11;
    } else if (d == 5) {
        if (beam_search) {
            vector<int> ef_coeff_{7, 10, 15, 22, 25, 30};
            ef_coeff.insert(ef_coeff.end(), ef_coeff_.begin(), ef_coeff_.end());
        } else {
            vector<int> k_coeff_{15, 20, 25, 30, 40, 60};
            k_coeff.insert(k_coeff.end(), k_coeff_.begin(), k_coeff_.end());
            vector<float> thr_coeff_{1.1, 1.15, 1.2, 1.3, 1.4, 1.5};
            thr_coeff.insert(thr_coeff.end(), thr_coeff_.begin(), thr_coeff_.end());
        }
        hops_bound = 7;
    } else if (d == 9) {
        if (beam_search) {
            vector<int> ef_coeff_{5, 8, 15, 25, 30, 35};
            ef_coeff.insert(ef_coeff.end(), ef_coeff_.begin(), ef_coeff_.end());
        } else {
            vector<int> k_coeff_{60, 100, 150, 200, 250, 300};
            k_coeff.insert(k_coeff.end(), k_coeff_.begin(), k_coeff_.end());
            vector<float> thr_coeff_{1.25, 1.3, 1.35, 1.4, 1.45, 1.5};
            thr_coeff.insert(thr_coeff.end(), thr_coeff_.begin(), thr_coeff_.end());
        }
        hops_bound = 5;
    } else if (d == 17) {
        if (beam_search) {
            vector<int> ef_coeff_{10, 40, 70, 100, 130, 160};
            ef_coeff.insert(ef_coeff.end(), ef_coeff_.begin(), ef_coeff_.end());
        } else {
            vector<int> k_coeff_{750, 1000, 1250, 1500, 1750, 2000};
            k_coeff.insert(k_coeff.end(), k_coeff_.begin(), k_coeff_.end());
            vector<float> thr_coeff_{1.1, 1.15, 1.17, 1.19, 1.21, 1.22};
            thr_coeff.insert(thr_coeff.end(), thr_coeff_.begin(), thr_coeff_.end());
        }
        hops_bound = 4;
    }

    int exp_size = min(ef_coeff.size(), k_coeff.size());

    for (int i=0; i < exp_size; ++i) {
        vector< vector <uint32_t>> knn_cur;
        if (beam_search) {
            knn_cur = knn;
        } else if (knn_by_threshold) {
            float thr = asin(thr_coeff[i] * pow(2, 0.5) * pow(n, - 1. / d));
            knn_cur = CutKNNbyThreshold(knn, db, thr, n, d, metric);
//            cout << "threshold " << thr << ", thr_coeff[i] " << thr_coeff[i] << endl;
        } else {
            knn_cur = CutKNNbyK(knn, db, k_coeff[i], n, d, metric);
        }

//        cout << "knn_cur  " << FindGraphAverageDegree(knn_cur) << endl;
        get_one_test(knn_cur, kl, db, queries, db, queries, truth, n, d, d, n_q, n_tr, ef_coeff[i], 1,
                     graph_name, metric, output_txt, inter_points, use_second_graph, llf, hops_bound, 0, recheck_size, 1, omp_get_max_threads());
    }

}
