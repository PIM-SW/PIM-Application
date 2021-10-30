# PIM-Application

## SVM-RFE
A classification benchmark from [Minebench: A Benchmark Suite for Data Mining Workloads](https://ieeexplore.ieee.org/document/4086147). Gene expression classifier using recursive feature elimination.

### How to Run
```bash
# Set OMP_NUM_THREADS
$ make
$ ./svm_mkl <dataset> <sample count> <gene count> <iteration count>
```

### Dataset
* base.txt: Ovarian cancer samples / 253 15154 30

### Profiling
* base: base.txt

## Reduction
A basic vector reduction kernel used in recommendation systems (Embedding Reduction) and graph neural networks (Feature Aggregation).