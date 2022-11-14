CXX=g++-9
CXXFLAGS=-I. -g -O3 -W -std=c++17 -lstdc++fs -fopenmp  -march=native

all: eval_baseline eval_remapped
eval_baseline: src/eval_baseline.cc
	$(CXX) -o bin/eval_baseline $< $(CXXFLAGS)
eval_remapped: src/eval_remapped.cc
	$(CXX) -o bin/eval_remapped $< $(CXXFLAGS)
