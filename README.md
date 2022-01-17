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

## EmbeddingReduction
A basic vector reduction kernel used in recommendation systems (Embedding Reduction) and graph neural networks (Feature Aggregation).

### How to Run
```bash
# install torch
$ pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
# install Ninja
$ sudo apt-get install ninja-build
# install mkl
$ sudo pip3 install mkl
# mlperf-loggin (optional)
$ git clone https://github.com/mlperf/logging.git mlperf-logging
$ pip install -e mlperf-logging
# run as RMC2 config
$ ./run.sh <pooling size> <query count>
```