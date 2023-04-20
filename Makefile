CUDA_INSTALL_PATH = /usr/local/cuda-11.8

#compilers
CC=$(CUDA_INSTALL_PATH)/bin/nvcc

#GLOBAL_PARAMETERS
MAT_VAL_TYPE = float
VALUE_TYPE = float

#CUDA_PARAMETERS
NVCC_FLAGS = -O3 -w -arch=compute_61 -code=sm_61 -gencode=arch=compute_61,code=sm_61
#-gencode=arch=compute_61,code=sm_75
# -m64 -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61
#-Xcompiler -Wall -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr

#ENVIRONMENT_PARAMETERS

#includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include

#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64  -lcudart  -lcusparse
LIBS = $(CUDA_LIBS)

#options
#OPTIONS = -std=c99

make:
	$(CC) $(NVCC_FLAGS) -Xcompiler -fopenmp -Xcompiler -mfma main-deepwalk.cu -o test $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D MAT_VAL_TYPE=$(MAT_VAL_TYPE)

deepwalk:
	$(CC) $(NVCC_FLAGS) -Xcompiler -fopenmp -Xcompiler -mfma main-deepwalk.cu -o deepwalk $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D MAT_VAL_TYPE=$(MAT_VAL_TYPE)

deepwalk-single:
	$(CC) $(NVCC_FLAGS) -Xcompiler -fopenmp -Xcompiler -mfma main-deepwalk-single.cu -o deepwalk-single $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D MAT_VAL_TYPE=$(MAT_VAL_TYPE)

sage:
	$(CC) $(NVCC_FLAGS) -Xcompiler -fopenmp -Xcompiler -mfma main-sage.cu -o sage $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D MAT_VAL_TYPE=$(MAT_VAL_TYPE)

sage-single:s
	$(CC) $(NVCC_FLAGS) -Xcompiler -fopenmp -Xcompiler -mfma main-sage-single.cu -o sage-single $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D MAT_VAL_TYPE=$(MAT_VAL_TYPE)

node2vec:
	$(CC) $(NVCC_FLAGS) -Xcompiler -fopenmp -Xcompiler -mfma main-node2vec.cu -o node2vec $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D MAT_VAL_TYPE=$(MAT_VAL_TYPE)
