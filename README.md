# Graph-based Nearest Neighbor Search: From Practice to Theory
Code for reproducing synthetic experiments from ICML2020 [paper](https://arxiv.org/abs/1907.00845)


#### Files description
`major_test.cpp` is code for reproducing main illustration of how analyzed heuristics and proposed methods work with synthetic datasets. 

Code from `links_type.cpp` analyze the effect of kNN and KL approximation and `number_links_test.cpp` illustrate effect of the number of long-range edges.

You can find results in `results` folder, with `draw_results.ipynb` for transformation '.txt' files to pictures.

##### Data prepare
All program will build corresponding graphs and datasets if necessary.

#### What about run?
Most functions are simple and straightforward, therefore machine with more CPU (only supported now) preferred.

To run it you need to specify paths to prospective data location in corresponding file and do
`g++ -Ofast -std=c++11 -fopenmp -march=native -fpic -w -ftree-vectorize file_name_test.cpp -o some_test.exe` 
and
 `./some_test.exe 16`
