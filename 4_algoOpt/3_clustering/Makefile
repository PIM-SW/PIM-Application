CXX=g++-9
CXXFLAGS=-I. -g -O3 -W -std=c++17 -lstdc++fs -fopenmp  -march=native -ffast-math

clustering: src/clustering.cc
	$(CXX) -o bin/clustering $< $(CXXFLAGS)