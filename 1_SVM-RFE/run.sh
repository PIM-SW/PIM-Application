cp svm_baseline.cpp svm.cpp
make all

printf "\nRunning Baseline...\n" 
./svm_mkl ./Dataset/base.txt 253 15154 30

mv ranking.txt ranking_baseline.txt
rm -f svm.cpp
cp svm_offload.cpp svm.cpp
make all

printf "\nRunning PIM Offload...\n" 
./svm_mkl ./Dataset/base.txt 253 15154 30

mv ranking.txt ranking_offload.txt
rm -f svm.cpp

