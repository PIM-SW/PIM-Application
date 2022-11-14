.SUFFIXES:  .c .o .cpp
################OPTION###################
CCOMPILE = gcc
CPPCOMPILE = g++
COMPILEOPTION = -g -D_OPT -fopenmp -D_MKL
# INCLUDEDIR = -I/opt/intel/vtune/analyzer/include -I/opt/intel/mkl721/include
INCLUDEDIR = -I. -I/opt/intel/oneapi/mkl/latest/include # -I/home/arcuser/pimiitp/OpenBLAS
LIBHOME = 
LINK = g++
LINKOPTION = -g -fopenmp -Wl,-rpath=/opt/intel/oneapi/mkl/latest/lib/intel64 -o ./svm_mkl
LIBS = -lpthread -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core#-lopenblas
LIBDIRS =  -L/opt/intel/oneapi/mkl/latest/lib/intel64 #-L/home/arcuser/pimiitp/OpenBLAS
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
	rm -f $(OUTPUT) ./ranking_baseline.txt ./ranking_offload.txt
