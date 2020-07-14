## Graph-based Nearest Neighbor Search: From Practice to Theory
Code for reproducing synthetic experiments from ICML2020 [paper](https://proceedings.icml.cc/static/paper_files/icml/2020/1229-Paper.pdf)


#### Files description
`major_test.cpp` is code for reproducing the main illustration of how analyzed heuristics and proposed methods work with synthetic datasets. 

Code from `links_type.cpp` analyze the effect of kNN and KL approximation and `number_links_test.cpp` illustrate the effect of the number of long-range edges.

For reproducing distribution of the distance to the nearest neighbor use `draw_1nn_distr.cpp`

You can find results in `results` folder, with `draw_results.ipynb` for transformation '.txt' files to pictures.

##### Data prepare
All programs will build corresponding graphs and datasets if necessary.

#### What about running?
Most of the functions are simple and straightforward, therefore machine with more CPU (only supported now) preferred.

To run it you need to specify paths to prospective data location in corresponding file and do
`g++ -Ofast -std=c++11 -fopenmp -march=native -fpic -w -ftree-vectorize file_name_test.cpp -o some_test.exe` 
and
 `./some_test.exe 16`
