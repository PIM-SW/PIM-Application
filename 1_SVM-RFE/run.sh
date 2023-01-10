mkdir -p IR
printf "Compiling SVM-RFE with PIM BLAS Library...\n\n"
make all
printf "\nRunning SVM-RFE using PIM API...\n" 
./svm_mkl ./Dataset/base.txt 253 15154 30
