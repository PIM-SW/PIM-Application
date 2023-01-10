.SUFFIXES:  .c .o .cpp
################OPTION###################
CCOMPILE = gcc
CPPCOMPILE = g++
#COMPILEOPTION = -w -D_OPT -fopenmp -D_MKL
COMPILEOPTION = -w -D_OPT -fopenmp
#INCLUDEDIR = -I/opt/intel/vtune/analyzer/include -I/opt/intel/mkl721/include
#INCLUDEDIR = -I. -I/opt/intel/oneapi/mkl/latest/include
INCLUDEDIR = -I../common
LIBHOME =
LINK = g++
#LINKOPTION = -fopenmp -Wl,-rpath=/opt/intel/oneapi/mkl/latest/lib/intel64 -o ./svm_mkl
LINKOPTION = -fopenmp -o ./svm_mkl
#LIBS = -lpthread -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core
LIBS = -lpthread -lopenblas
#LIBDIRS =  -L/opt/intel/oneapi/mkl/latest/lib/intel64
LIBDIRS = -L../common
OBJS = wtime.o svm.o svm_rfe.o svm_rfe_test.o
OUTPUT = ./svm_mkl
################OPTION END################
all: $(OUTPUT)

$(OUTPUT):$(OBJS)
	$(LINK) $(OBJS) $(LINKOPTION) $(LIBS) $(LIBDIRS)


.cpp.o:
	 $(CPPCOMPILE) -c $(COMPILEOPTION) $(INCLUDEDIR) $<

.c.o:
	$(CCOMPILE) -c $(COMPILEOPTION) $(INCLUDEDIR) $<

clean: 
	rm -f $(OBJS)
	rm -f $(OUTPUT) ./ranking.txt
	rm -f ./IR/*.trace
