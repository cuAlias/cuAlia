#include "common.h"
#include "mmio_highlevel.h"
// #include"mmio.h"
#include "utils.h"
#include "utils_tile.h"
// #include"format_trans.h"
// #include"spmv_tile.h"
// #include"spmv_tile_balance.h"
// #include"tilespmv_warp_bal.h"

#include "LBLT.h"
#include "step.h"
// #include"spmv_cuda.h"
// #include <thrust/sort.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
typedef struct
{
  int m;
  int n;
  int nnz;
  MAT_VAL_TYPE *value;
  int *columnidx;
  MAT_PTR_TYPE *rowpointer;
  int isSymmetric;
  int *alias;
  MAT_VAL_TYPE *prob;

  int *small;
  int *large;
  MAT_VAL_TYPE *p;
} Matrix;

#define WARP_SIZE 32
#define WARP_PER_BLOCK 2

template <typename T>
__inline__ __device__ T warpReduce(T val)
{
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}
__global__ void AliasConstructKernelCUDA(
    int *alias,
    MAT_VAL_TYPE *prob,
    int *small_idx_ptr,
    int *large_idx_ptr,
    int *small,
    int *large,
    MAT_VAL_TYPE *p,
    MAT_PTR_TYPE *rowpointerA,
    int *columnindexA,
    MAT_VAL_TYPE *value,
    int num_nodes,
    int nnz)
{

  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  int input_idx = gidx / 32;
  int lid = gidx % 32;
  if (input_idx > num_nodes - 1)
    return;

  //   PtrGen<WMOffsetType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
  //   PtrGen<WMIdType, IdType> csr_col_ptr_gen(wm_csr_col_ptr);
  //   PtrGen<WMWeightType, WeightType> csr_weight_ptr_gen(wm_csr_weight_ptr);
  //   int64_t start = *csr_row_ptr_gen.At(input_idx);
  //   int64_t end = *csr_row_ptr_gen.At(input_idx + 1);

  int start = rowpointerA[input_idx];
  int end = rowpointerA[input_idx + 1];
  int neighbor_count = (int)(end - start);

  MAT_VAL_TYPE local_sum = 0.0, tmp, weight_sum;
  for (int i = lid; i < neighbor_count; i += 32)
  {
    //  MAT_VAL_TYPE thread_weight = *(csr_weight_ptr_gen.At(start + i));
    MAT_VAL_TYPE thread_weight = value[start + i];
    local_sum += thread_weight;
  }
  tmp = warpReduce<MAT_VAL_TYPE>(local_sum);
  weight_sum = __shfl_sync(0xffffffff, tmp, 0, 32);
  if (weight_sum != 0.0)
  {
    MAT_VAL_TYPE scale = (MAT_VAL_TYPE)neighbor_count / weight_sum;
    for (int i = lid; i < neighbor_count; i += 32)
    {
      // p[start+i] = *(csr_weight_ptr_gen.At(start + i)) * scale;
      p[start + i] = value[start + i] * scale;
    }
  }
  else
  {
    // printf("weight_sum = 0.0!\n");
    return;
  }

  small_idx_ptr[input_idx] = 0;
  large_idx_ptr[input_idx] = 0;
  for (int i = lid; i < neighbor_count; i += 32)
  {
    prob[start + i] = 1.0;
    alias[start + i] = 0;
  }
  __syncthreads();
  for (int i = lid; i < neighbor_count; i += 32)
  {
    MAT_VAL_TYPE tmp_weight = p[start + i];
    if (tmp_weight >= 1)
    {
      int k = atomicAdd(&large_idx_ptr[input_idx], 1);
      large[start + k] = i;
    }
    else
    {
      int k = atomicAdd(&small_idx_ptr[input_idx], 1);
      small[start + k] = i;
    }
  }

  __syncthreads();
  int tmp_small_idx = atomicSub(&small_idx_ptr[input_idx], 1);
  int tmp_large_idx = atomicSub(&large_idx_ptr[input_idx], 1);

  if (tmp_small_idx <= 0 || tmp_large_idx <= 0)
  {
    atomicAdd(&small_idx_ptr[input_idx], 1);
    atomicAdd(&large_idx_ptr[input_idx], 1);
  }

  while (tmp_small_idx > 0 && tmp_large_idx > 0)
  {
    tmp_small_idx--;
    tmp_large_idx--;
    if (tmp_small_idx >= 0 && tmp_large_idx >= 0) //&&flag==1
    {
      int s_idx = small[start + tmp_small_idx];
      int l_idx = large[start + tmp_large_idx];
      prob[start + s_idx] = p[start + s_idx];
      alias[start + s_idx] = l_idx;

      atomicAdd(&p[start + l_idx], p[start + s_idx] - 1);

      if (p[start + l_idx] <= 1)
      {
        tmp_small_idx = atomicAdd(&small_idx_ptr[input_idx], 1);
        small[start + tmp_small_idx] = l_idx;
      }
      else
      {
        tmp_large_idx = atomicAdd(&large_idx_ptr[input_idx], 1);
        large[start + tmp_large_idx] = l_idx;
      }
    }

    tmp_small_idx = atomicSub(&small_idx_ptr[input_idx], 1);
    tmp_large_idx = atomicSub(&large_idx_ptr[input_idx], 1);

    if (tmp_small_idx <= 0 || tmp_large_idx <= 0)
    {
      atomicAdd(&small_idx_ptr[input_idx], 1);
      atomicAdd(&large_idx_ptr[input_idx], 1);
    }
  }
}

__host__ __device__ __forceinline__ void RandomNum(int gid, unsigned long long seed, unsigned long long *next_random)
{

  *next_random = seed + gid;
  *next_random ^= *next_random >> 33U;
  *next_random *= 0xff51afd7ed558ccdUL;
  *next_random ^= *next_random >> 33U;
  *next_random *= 0xc4ceb9fe1a85ec53UL;
  *next_random ^= *next_random >> 33U;
}
__host__ __device__ __forceinline__ int Random(unsigned long long *next_random)
{
  int ret_value = (int)(*next_random & 0x7fffffffULL);
  *next_random = *next_random * (unsigned long long)13173779397737131ULL + 1023456798976543201ULL;
  return ret_value;
}

__host__ __device__ __forceinline__ float RandomUniformFloat(unsigned long long *next_random)
{
  float max = 1.0f, min = 0.0f;
  int value = (int)(*next_random & 0xffffff);
  auto ret_value = (float)value;
  ret_value /= 0xffffffL;
  ret_value *= (max - min);
  ret_value += min;
  *next_random = *next_random * (unsigned long long)13173779397737131ULL + 1023456798976543201ULL;
  return ret_value;
}

__global__ void AliasSampleKernel(int *output,
                                  // LocalIdType *src_lid,
                                  MAT_PTR_TYPE *sample_offset,
                                  int *input_nodes,
                                  int input_node_count,
                                  MAT_PTR_TYPE *rowpointerA,
                                  int *columnindexA,
                                  int max_sample_count,
                                  unsigned long long random_seed,
                                  int *alias,
                                  MAT_VAL_TYPE *prob)
{
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  // int tidx = gidx%32;
  int input_idx = blockIdx.x;
  if (input_idx > input_node_count - 1)
    return;
  // RandomNumGen rng(gidx, random_seed);
  // rng.NextValue();
  unsigned long long next_random = 1;
  RandomNum(gidx, random_seed, &next_random);
  Random(&next_random);
  //   whole_memory::PtrGen<WMOffsetType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
  //   whole_memory::PtrGen<WMIdType, IdType> csr_col_ptr_gen(wm_csr_col_ptr);
  //   whole_memory::PtrGen<WMIdType, IdType> alias_gen(wm_alias);
  //   whole_memory::PtrGen<WMWeightType, WeightType> prob_gen(wm_prob);

  int nid = input_nodes[input_idx];
  int start = rowpointerA[nid];
  int end = rowpointerA[nid + 1];
  int neighbor_count = (int)(end - start);
  int offset = sample_offset[input_idx];

  // sample all
  if (neighbor_count <= max_sample_count)
  {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count && sample_id < max_sample_count; sample_id += blockDim.x)
    {
      int neighbor_idx = sample_id;
      // IdType gid = *csr_col_ptr_gen.At(start + neighbor_idx);
      int gid = columnindexA[start + neighbor_idx];
      output[offset + sample_id] = gid;
      // printf("cuda : %d %d\n",input_idx,gid);
      // if (src_lid) src_lid[offset + sample_id] = (LocalIdType) input_idx;
    }
    return;
  }

  for (int sample_id = threadIdx.x; sample_id < max_sample_count; sample_id += blockDim.x)
  {

    int col = Random(&next_random) % neighbor_count;
    MAT_VAL_TYPE p = RandomUniformFloat(&next_random);

    // if (p > prob[(int)start+col])
    // if(p < *prob_gen.At(start+col))
    if (p <= prob[start + col])
    {
      int neighbor_idx = col;
      // IdType gid = *csr_col_ptr_gen.At(start + neighbor_idx);
      int gid = columnindexA[start + neighbor_idx];
      output[offset + sample_id] = gid;
    }
    else
    {

      // int neighbor_idx = *alias_gen.At(start+col);
      int neighbor_idx = alias[start + col];
      // IdType gid = *csr_col_ptr_gen.At(start + neighbor_idx);
      int gid = columnindexA[start + neighbor_idx];
      output[offset + sample_id] = gid;
    }

    // if (src_lid) src_lid[offset + sample_id] = (LocalIdType) input_idx;
  }
}

int main(int argc, char **argv)
{
  printf("--------------------------------!!-cuda-!!------------------------------------\n");
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
  printf("dev: %s\n",deviceProp.name);
  char *filename;
  filename = argv[1];

  printf("MAT: -------------- %s --------------\n", filename);
  Matrix *matrixA_d = (Matrix *)malloc(sizeof(Matrix));
  struct timeval t1, t2;

  // load mtx A data to the csr format
  gettimeofday(&t1, NULL);
  mmio_allinone(&matrixA_d->m, &matrixA_d->n, &matrixA_d->nnz, &matrixA_d->isSymmetric, &matrixA_d->rowpointer, &matrixA_d->columnidx, &matrixA_d->value, filename);
  gettimeofday(&t2, NULL);
  double time_loadmat = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
  printf("input matrix A: ( %i, %i ) nnz = %i\n loadfile time    = %4.5f sec\n", matrixA_d->m, matrixA_d->n, matrixA_d->nnz, time_loadmat / 1000.0);

  for (int i = 0; i < matrixA_d->nnz; i++)
    matrixA_d->value[i] = i % 10 + 1;

  MAT_PTR_TYPE *d_rowpointerA;
  int *d_columnindexA;
  MAT_VAL_TYPE *d_value;

  cudaMalloc((void **)&d_rowpointerA, sizeof(MAT_PTR_TYPE) * (matrixA_d->m + 1));
  cudaMemcpy(d_rowpointerA, matrixA_d->rowpointer, sizeof(MAT_PTR_TYPE) * (matrixA_d->m + 1), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_columnindexA, sizeof(int) * (matrixA_d->nnz));
  cudaMemcpy(d_columnindexA, matrixA_d->columnidx, sizeof(int) * (matrixA_d->nnz), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_value, (matrixA_d->nnz) * sizeof(MAT_VAL_TYPE));
  cudaMemcpy(d_value, matrixA_d->value, (matrixA_d->nnz) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

  int sample_size = atoi(argv[2]);
  int *input_nodes = (int *)malloc(sizeof(int) * sample_size);
  printf("sample_size : %d\n",sample_size);
    
  int max_rowid=0;
  int max_rowlen=0;
  for(int i=0;i<matrixA_d->m+1;i++)
  {
    int len=matrixA_d->rowpointer[i+1]-matrixA_d->rowpointer[i];
    if(len>max_rowlen)
    {
      max_rowlen=len;
      max_rowid=i;
    }
  }
   printf("%d len :%d\n",max_rowid,max_rowlen);
   
  int act_max_rowid=0;
  int act_max_rowlen=0;
  //srand((unsigned)time(NULL));
  srand(1);
  for (int i = 0; i < sample_size; i++)
  {
    input_nodes[i] = (rand() * (unsigned long long)13173779397737131ULL + 1023456798976543201ULL) % matrixA_d->m;
    // printf("%d\n",rand()%100+1);
    int len=matrixA_d->rowpointer[input_nodes[i]+1]-matrixA_d->rowpointer[input_nodes[i]];
    if(len>act_max_rowlen)
    {
      act_max_rowlen=len;
      act_max_rowid=input_nodes[i];
    }
  }
  printf("%d act_len :%d\n",act_max_rowid,act_max_rowlen);
  unsigned long long random_seed = rand() * (unsigned long long)13173779397737131ULL + 1023456798976543201ULL;
  int *d_input_nodes;
  cudaMalloc((void **)&d_input_nodes, sizeof(int) * (sample_size));
  cudaMemcpy(d_input_nodes, input_nodes, sizeof(int) * (sample_size), cudaMemcpyHostToDevice);

  //----------------------Alias Table Construct---------------------------
  int sample_count = 10; // 单个点采样数量

  MAT_PTR_TYPE *sample_offset = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (sample_size + 1));
  for (int i = 0; i < sample_size; i++)
  {
    MAT_PTR_TYPE vid = input_nodes[i];
    int neighbor_count = (int)(matrixA_d->rowpointer[vid + 1] - matrixA_d->rowpointer[vid]);
    sample_offset[i] = min(neighbor_count, sample_count);
  }
  exclusive_scan(sample_offset, sample_size + 1);
  MAT_PTR_TYPE *d_sample_offset;
  cudaMalloc((void **)&d_sample_offset, sizeof(MAT_PTR_TYPE) * (sample_size + 1));
  cudaMemcpy(d_sample_offset, sample_offset, sizeof(MAT_PTR_TYPE) * (sample_size + 1), cudaMemcpyHostToDevice);
  MAT_PTR_TYPE sample_offset_sum = sample_offset[sample_size]; // 所有点实际采样数量
  // printf("")
  // MAT_PTR_TYPE *small_idx_ptr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (matrixA_d->m) );
  // memset(small_idx_ptr, 0, sizeof(MAT_PTR_TYPE) * (matrixA_d->m));
  int *d_small_idx_ptr;
  cudaMalloc((void **)&d_small_idx_ptr, sizeof(int) * (matrixA_d->m));
  cudaMemset(d_small_idx_ptr, 0, sizeof(int) * (matrixA_d->m));

  int *d_large_idx_ptr;
  cudaMalloc((void **)&d_large_idx_ptr, sizeof(int) * (matrixA_d->m));
  cudaMemset(d_large_idx_ptr, 0, sizeof(int) * (matrixA_d->m));

  int *d_large;
  cudaMalloc((void **)&d_large, sizeof(int) * (matrixA_d->nnz));
  cudaMemset(d_large, 0, sizeof(int) * (matrixA_d->nnz));

  int *d_small;
  cudaMalloc((void **)&d_small, sizeof(int) * (matrixA_d->nnz));
  cudaMemset(d_small, 0, sizeof(int) * (matrixA_d->nnz));

  MAT_VAL_TYPE *d_p;
  cudaMalloc((void **)&d_p, sizeof(MAT_VAL_TYPE) * (matrixA_d->nnz));
  cudaMemset(d_p, 0, sizeof(MAT_VAL_TYPE) * (matrixA_d->nnz));

  int *d_alias;
  cudaMalloc((void **)&d_alias, sizeof(int) * (matrixA_d->nnz));
  cudaMemset(d_alias, 0, sizeof(int) * (matrixA_d->nnz));

  MAT_VAL_TYPE *d_prob;
  cudaMalloc((void **)&d_prob, sizeof(MAT_VAL_TYPE) * (matrixA_d->nnz));
  cudaMemset(d_prob, 0, sizeof(MAT_VAL_TYPE) * (matrixA_d->nnz));

  /*  int *d_level_loc;
    cudaMalloc((void **)&d_level_loc, sizeof(int) * (matrixA_d->m) );
    cudaMemset(d_level_loc, 0, sizeof(int) * (matrixA_d->m));

    MAT_VAL_TYPE *d_level_prob;
    cudaMalloc((void **)&d_level_prob, sizeof(MAT_VAL_TYPE) * (matrixA_d->m) );
    cudaMemset(d_level_prob, 0, sizeof(MAT_VAL_TYPE) * (matrixA_d->m));
*/
  cudaDeviceSynchronize();
  gettimeofday(&t1, NULL);
  AliasConstructKernelCUDA<<<matrixA_d->m, 32>>>(d_alias,
                                                 d_prob,
                                                 d_small_idx_ptr,
                                                 d_large_idx_ptr,
                                                 d_small,
                                                 d_large,
                                                 d_p,
                                                 d_rowpointerA,
                                                 d_columnindexA,
                                                 d_value,
                                                 matrixA_d->m,
                                                 matrixA_d->nnz);

  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  double time_alias_table_constrcut = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
  printf("Alias Table Construct: %f ms\n", time_alias_table_constrcut);

  cudaFree(d_small_idx_ptr);
  cudaFree(d_large_idx_ptr);
  cudaFree(d_large);
  cudaFree(d_small);
  cudaFree(d_p);
  //----------------------Alias Table Construct End---------------------------
  // matrixA_d->alias = (int *)malloc(sizeof(int)*matrixA_d->nnz);
  // matrixA_d->prob = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE)*matrixA_d->nnz);
  // cudaMemcpy(matrixA_d->alias, d_alias, sizeof(int) * (matrixA_d->nnz), cudaMemcpyDeviceToHost);
  // cudaMemcpy(matrixA_d->prob, d_prob, sizeof(MAT_VAL_TYPE) * (matrixA_d->nnz), cudaMemcpyDeviceToHost);
  // for(int i=0;i<matrixA_d->m;i++)
  // {
  //   for(int j=matrixA_d->rowpointer[i];j<matrixA_d->rowpointer[i+1];j++)
  //   {
  //     printf("%d  ",matrixA_d->columnidx[j]);
  //   }
  //   printf("\n");
  //   for(int j=matrixA_d->rowpointer[i];j<matrixA_d->rowpointer[i+1];j++)
  //   {
  //     printf("%lf  ",matrixA_d->value[j]);
  //   }
  //   printf("\n");
  //   for(int j=matrixA_d->rowpointer[i];j<matrixA_d->rowpointer[i+1];j++)
  //   {
  //     printf("%d  ",matrixA_d->alias[j]);
  //   }
  //   printf("\n");
  //   for(int j=matrixA_d->rowpointer[i];j<matrixA_d->rowpointer[i+1];j++)
  //   {
  //     printf("%lf  ",matrixA_d->prob[j]);
  //   }
  //   printf("\n");
  //   printf("\n");
  // }
  // printf("\n");
  //----------------------Alias Table Sample---------------------------

  int *sample_output = (int *)malloc(sizeof(int) * (sample_offset_sum));

  int *d_sample_output;
  cudaMalloc((void **)&d_sample_output, sizeof(int) * sample_offset_sum);
  double ave_time_alias_table_sample = 0;
  for(int iter = 0; iter < 1000; iter++)
  {
    cudaMemset(d_sample_output, 0, sizeof(int) * sample_offset_sum);
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    AliasSampleKernel<<<sample_size, 64>>>(d_sample_output,
                                         // src_lid,
                                         d_sample_offset,
                                         d_input_nodes,
                                         sample_size,
                                         d_rowpointerA,
                                         d_columnindexA,
                                         sample_count,
                                         random_seed,
                                         d_alias,
                                         d_prob);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time_alias_table_sample = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    ave_time_alias_table_sample += time_alias_table_sample;
  }
  ave_time_alias_table_sample=ave_time_alias_table_sample/1000;
  
  printf("Alias Table Sample: %f ms\n", ave_time_alias_table_sample);

  // FILE *fout = fopen("10-20%-results-4090.csv", "a");
  //   if (fout == NULL)
  //       printf("Writing results fails.\n");
  //   fprintf(fout, "%s,%i,%i,%i,%i,%i,%f,%f\n",
  //           filename,  matrixA_d->m, matrixA_d->n, matrixA_d->nnz,max_rowid,max_rowlen, time_alias_table_constrcut, ave_time_alias_table_sample);
  //   fclose(fout);

  cudaMemcpy(sample_output, d_sample_output, sizeof(int) * (sample_offset_sum), cudaMemcpyDeviceToHost);

  // for(int i=0;i<sample_size;i++)
  // {
  //   printf("%d : ",input_nodes[i]);
  //   for(int j=matrixA_d->rowpointer[input_nodes[i]];j<matrixA_d->rowpointer[input_nodes[i]+1];j++)
  //   {
  //     printf("%d  ",matrixA_d->columnidx[j]);
  //   }
  //   printf("\n");
  //   for(int j=matrixA_d->rowpointer[input_nodes[i]];j<matrixA_d->rowpointer[input_nodes[i]+1];j++)
  //   {
  //     printf("%lf  ",matrixA_d->value[j]);
  //   }
  //   printf("\n");
  //   for(int j=sample_offset[i];j<sample_offset[i+1];j++)
  //   {
  //     printf("%d  ",sample_output[j]);
  //   }

  //   printf("\n");
  //   printf("\n");
  // }
  // printf("%d\n",sample_offset_sum);
  // for(int i=0;i<sample_offset_sum;i++)
  //  printf("%d  ",sample_output[i]);
  //----------------------Alias Table Sample End---------------------------
  cudaFree(d_alias);
  cudaFree(d_prob);
  cudaFree(d_rowpointerA);
  cudaFree(d_columnindexA);
  cudaFree(d_value);

  free(input_nodes);
  free(matrixA_d->rowpointer);
  free(matrixA_d->columnidx);
  free(matrixA_d->value);

  return 0;
}
/* FILE *fout = fopen("results.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%i,%i,%i,%f\n",
            filename,  matrixA_d->m, matrixA_d->n, matrixA_d->nnz, matrixA_d->time_cuda_step);
    fclose(fout);
    */
