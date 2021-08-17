.SUFFIXES:  .c .o .cpp
################OPTION###################
CCOMPILE = gcc
CPPCOMPILE = g++
COMPILEOPTION = -g -fopenmp -D_OPT
# INCLUDEDIR = -I/opt/intel/vtune/analyzer/include -I/opt/intel/mkl721/include
INCLUDEDIR = -I. -I/home/arcuser/pimiitp/OpenBLAS -I/opt/intel/oneapi/mkl/latest/include
LIBHOME = 
LINK = g++
LINKOPTION = -g -fopenmp -o ./svm_mkl
LIBS =  /opt/intel/mkl721/lib/32/libmkl_ia32.a
LIBS =  -lopenblas -lpthread
LIBDIRS =  -L/home/arcuser/pimiitp/OpenBLAS
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
	rm -f $(OUTPUT)
