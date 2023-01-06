printf "Compiling SVM-RFE...\n"
make all
printf "\nRunning SVM-RFE...\n" 
./svm_mkl ./Dataset/base.txt 253 15154 30