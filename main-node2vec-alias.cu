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
#include <curand_kernel.h>
#include <queue>

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


__global__ void Node2vecKernel(int *output,//采样结果
  MAT_PTR_TYPE *sample_offset,//采样个数前缀和
  int *input_nodes,//sample node
  int input_node_count,
  MAT_PTR_TYPE *rowpointerA,
  int *columnindexA,
  MAT_VAL_TYPE *valA,
  //int max_sample_count,//25
  int length,//采样深度
  unsigned long long random_seed,
  float *alpha,
  int *alias,
  MAT_VAL_TYPE *prob,
  int *pre_edge
)
{
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;//thread id
  // int widx = gidx / 32;//warp id
  // int tidx = gidx % 32;//tread id in warp
  curandState state;
  curand_init(gidx, 0, 0, &state);

  int input_idx = gidx;//采样起始点
  if (input_idx > input_node_count - 1)
  return;

  int nid = input_nodes[input_idx];
  for(int dep=0;dep<length;dep++) {
      
      int start = rowpointerA[nid];
      int end = rowpointerA[nid + 1];
      int neighbor_count = (int)(end - start);
      int offset = sample_offset[input_idx];
      int pre_eid;//前一个采样边

      //度=0 结束
      if(neighbor_count==0) return;

      if(neighbor_count==1) {//只有一条边
        int gid = columnindexA[start];
        output[offset + dep] = gid;
        nid=gid;
        pre_eid=start;
      } 
      else {//边数 > 1
        if(dep==0) {
          for(int i=start;i<end;i++) {
            alpha[i]=(float)valA[i];
          }
          for(int i=1;i<neighbor_count;i++) alpha[start+i]+=alpha[start+i-1];
          float p=curand_uniform(&state)*alpha[start+neighbor_count-1];
          int l=0,r=neighbor_count-1;
          int res=0;
          while(l<=r) {
            int mid=(l+r)/2;
            float lval,rval;
            if(mid==0) {
              lval=0;
              rval=alpha[start];
            }
            else {
              lval=alpha[start+mid-1];
              rval=alpha[start+mid];
            }
            if(p>=lval&&p<=rval) {
              res=mid;
              break;
            }
            if(lval>p) r=mid-1;
            else l=mid+1;
          }
          int gid = columnindexA[start+res];
          output[offset + dep] = gid;
          nid = gid;
          pre_eid=start+res;
        }
        else {
          int col = (int)floor(curand_uniform(&state) * neighbor_count);
          float p = curand_uniform(&state);
          int pre=pre_edge[pre_eid];
          if (p <= prob[pre + col])
          {
            int neighbor_idx = col;
            // IdType gid = *csr_col_ptr_gen.At(start + neighbor_idx);
            int gid = columnindexA[start + neighbor_idx];
            output[offset + dep] = gid;
            pre_eid=start+neighbor_idx;
            nid=gid;
          }
          else
          {
      
            // int neighbor_idx = *alias_gen.At(start+col);
            int neighbor_idx = alias[pre + col];
            // IdType gid = *csr_col_ptr_gen.At(start + neighbor_idx);
            int gid = columnindexA[start + neighbor_idx];
            output[offset + dep] = gid;
            pre_eid=start+neighbor_idx;
            nid=gid;
          }
        }
      }
  }
}


__global__ void warmup() {
  int sum=0;
  for(int i=0;i<1000;i++) sum+=1;
}

bool check(Matrix *matrix,int x,int y) {
  int l=matrix->rowpointer[x],r=matrix->rowpointer[x+1]-1;
  while(l<=r) {
    int mid=(l+r)/2;
    int cur=matrix->columnidx[mid];
    if(cur==y) return true;
    if(cur>y) r=mid-1;
    else l=mid+1;
  }
  return false;
}

void construct_aliastable(int *alias,double *prob,double *b,Matrix *matrix,int *pre_edge) {
  for(int i=0;i<matrix->nnz;i++) {//第i条边
    int pre=pre_edge[i];//bias 
    int x=matrix->columnidx[i];//指向的点x
    int n=matrix->rowpointer[x+1]-matrix->rowpointer[x];//x的邻边
    double sum=0;
    for(int j=0;j<n;j++)  sum+=b[pre+j];
    
    for(int j=0;j<n;j++) prob[pre+j]=n*b[pre+j]/sum;

    std::queue<int> large,small;
    for(int j=0;j<n;j++) {
      if(prob[pre+j]>1) large.push(j);
      else small.push(j);
    }

    while(!large.empty()&&!small.empty()) {
      int vl=large.front();
      large.pop();
      int vs=small.front();
      small.pop();
      prob[pre+vl]=prob[pre+vl]+prob[pre+vs]-1;
      alias[pre+vs]=vl;
      if(prob[pre+vl]>1) large.push(vl);
      else if(prob[pre+vl]<1) small.push(vl);
    }
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

  // for (int i = 0; i < matrixA_d->nnz; i++)
  //   matrixA_d->value[i] = i % 10 + 1;

  MAT_PTR_TYPE *d_rowpointerA;
  int *d_columnindexA;
  MAT_VAL_TYPE *d_value;
  //复制到device
  cudaMalloc((void **)&d_rowpointerA, sizeof(MAT_PTR_TYPE) * (matrixA_d->m + 1));
  cudaMemcpy(d_rowpointerA, matrixA_d->rowpointer, sizeof(MAT_PTR_TYPE) * (matrixA_d->m + 1), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_columnindexA, sizeof(int) * (matrixA_d->nnz));
  cudaMemcpy(d_columnindexA, matrixA_d->columnidx, sizeof(int) * (matrixA_d->nnz), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_value, (matrixA_d->nnz) * sizeof(MAT_VAL_TYPE));
  cudaMemcpy(d_value, matrixA_d->value, (matrixA_d->nnz) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

  int sample_size = atoi(argv[2]);
  int *input_nodes = (int *)malloc(sizeof(int) * sample_size);
  printf("sample_size : %d\n",sample_size);
    
  //记录长行
  int max_rowid=0;
  int max_rowlen=0;
  for(int i=0;i<matrixA_d->m;i++)
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
    // input_nodes[i]=i;
    //获取采样起始点
    // printf("%d\n",input_nodes[i]);
    int len=matrixA_d->rowpointer[input_nodes[i]+1]-matrixA_d->rowpointer[input_nodes[i]];
    if(len>act_max_rowlen)
    {
      act_max_rowlen=len;
      act_max_rowid=input_nodes[i];
    }
  }
  printf("%d act_len :%d\n",act_max_rowid,act_max_rowlen);
  unsigned long long random_seed = rand() * (unsigned long long)13173779397737131ULL + 1023456798976543201ULL;
  //采样点复制到device
  int *d_input_nodes;
  cudaMalloc((void **)&d_input_nodes, sizeof(int) * (sample_size));
  cudaMemcpy(d_input_nodes, input_nodes, sizeof(int) * (sample_size), cudaMemcpyHostToDevice);

  //----------------------构建alias table---------------------------
  gettimeofday(&t1, NULL);

  //获取每一条边->node数量
  int edge_node_sum = 0;
  int *pre_edge = (int*)malloc(sizeof(int)*(matrixA_d->nnz+1));
  for(int i=0;i<matrixA_d->m;i++) {
    for(int j=matrixA_d->rowpointer[i];j<matrixA_d->rowpointer[i+1];j++) {
      int y=matrixA_d->columnidx[j];
      int edge_num=matrixA_d->rowpointer[y+1]-matrixA_d->rowpointer[y];
      pre_edge[j]=edge_num;
    }
  }
  exclusive_scan(pre_edge, matrixA_d->nnz + 1);
  edge_node_sum = pre_edge[matrixA_d->nnz];

  int *d_pre_edge;
  cudaMalloc((void **)&d_pre_edge, sizeof(int) * (matrixA_d->nnz+1));
  cudaMemcpy(d_pre_edge, pre_edge, sizeof(int) * (matrixA_d->nnz+1), cudaMemcpyHostToDevice);

  double *b = (double*)malloc(sizeof(double)*edge_node_sum);
  double p=2.0,q=0.5;
  for(int i=0;i<matrixA_d->m;i++) {
    for(int j=matrixA_d->rowpointer[i];j<matrixA_d->rowpointer[i+1];j++) {
      int y=matrixA_d->columnidx[j];
      for(int k=matrixA_d->rowpointer[y];k<matrixA_d->rowpointer[y+1];k++) {
        int x=matrixA_d->columnidx[k];
        int w=matrixA_d->value[k];
        int cur_id=pre_edge[j] + k - matrixA_d->rowpointer[y];
        
        if(i==x) b[cur_id]=1/p;//x是i
        else if(check(matrixA_d,i,x)) b[cur_id]=1;//x是i的邻边
        else b[cur_id]=1/q;
        
        b[cur_id]*=w;
      }
    }
  }

  int *alias = (int*)malloc(sizeof(int)*edge_node_sum);
  double *prob = (double*)malloc(sizeof(double)*edge_node_sum);
  construct_aliastable(alias,prob,b,matrixA_d,pre_edge);

  gettimeofday(&t2, NULL);
  double time_alias_table_create = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
  printf("Create Alias Table: %f ms\n", time_alias_table_create);

  // printf("output w:\n");
  // for(int i=0;i<matrixA_d->m;i++) {
  //   for(int j=matrixA_d->rowpointer[i];j<matrixA_d->rowpointer[i+1];j++) {
  //     int y=matrixA_d->columnidx[j];
  //     printf("edge x:%d to y:%d = %lf\n",i,y,matrixA_d->value[j]);
  //   }
  // }  

  // printf("output aliastable:\n");
  // for(int i=0;i<matrixA_d->m;i++) {
  //   for(int j=matrixA_d->rowpointer[i];j<matrixA_d->rowpointer[i+1];j++) {
  //     int y=matrixA_d->columnidx[j];
  //     printf("edge x:%d to y:%d:\n",i,y);
  //     for(int k=matrixA_d->rowpointer[y];k<matrixA_d->rowpointer[y+1];k++) {
  //       int id=pre_edge[j]+k-matrixA_d->rowpointer[y];
  //       printf("b=%lf, alias = %d, prob = %lf,\n",b[id],alias[id],prob[id]);
  //     }
  //     printf("\n");
  //   }
  // }

  MAT_PTR_TYPE *d_alias;
  cudaMalloc((void **)&d_alias, sizeof(MAT_PTR_TYPE) * edge_node_sum);
  cudaMemcpy(d_alias, alias, sizeof(MAT_PTR_TYPE) * edge_node_sum, cudaMemcpyHostToDevice);
  
  MAT_VAL_TYPE *d_prob;
  cudaMalloc((void **)&d_prob, sizeof(MAT_VAL_TYPE) * (edge_node_sum));
  cudaMemcpy(d_prob, prob, sizeof(MAT_VAL_TYPE) * (edge_node_sum), cudaMemcpyHostToDevice);

  //----------------------获取采样结果前缀和---------------------------
  int deep = 100; // 单个点采样数量、深度

  MAT_PTR_TYPE *sample_offset = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (sample_size + 1));
  for (int i = 0; i < sample_size; i++)
  {
    sample_offset[i] = deep;
  }
  //前缀和
  exclusive_scan(sample_offset, sample_size + 1);
  MAT_PTR_TYPE *d_sample_offset;
  cudaMalloc((void **)&d_sample_offset, sizeof(MAT_PTR_TYPE) * (sample_size + 1));
  cudaMemcpy(d_sample_offset, sample_offset, sizeof(MAT_PTR_TYPE) * (sample_size + 1), cudaMemcpyHostToDevice);
  MAT_PTR_TYPE sample_offset_sum = sample_offset[sample_size]; // 所有点实际采样数量
  // printf("")
  // MAT_PTR_TYPE *small_idx_ptr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (matrixA_d->m) );
  // memset(small_idx_ptr, 0, sizeof(MAT_PTR_TYPE) * (matrixA_d->m));


  //----------------------Alias Table Sample---------------------------

  int *sample_output = (int *)malloc(sizeof(int) * (sample_offset_sum));//大小是采样结果个数

  int *d_sample_output;
  cudaMalloc((void **)&d_sample_output, sizeof(int) * sample_offset_sum);
  double ave_time_alias_table_sample = 0;//采样时间
  //采样结果
  cudaMemset(d_sample_output, 0, sizeof(int) * sample_offset_sum);
  //alpha_pq(t,x)
  float *d_alpha;
  cudaMalloc((void **)&d_alpha, sizeof(float) * (matrixA_d->nnz));
  //预热
  warmup<<<sample_size,64>>>();
  cudaDeviceSynchronize();
  
  gettimeofday(&t1, NULL);
  for(int iter = 0; iter < 1000; iter++)
  {
    //每个warp处理一个sample node
    //每个block包含4个warp，处理4个sample node 
    Node2vecKernel<<<sample_size/128 + 1, 128>>>(d_sample_output,//采样结果
                                         // src_lid,
                                         d_sample_offset,//采样个数前缀和
                                         d_input_nodes,//sample node
                                         sample_size,
                                         d_rowpointerA,
                                         d_columnindexA,
                                         d_value,
                                         deep,
                                         random_seed,
                                         d_alpha,
                                         d_alias,
                                         d_prob,
                                         d_pre_edge);
    cudaDeviceSynchronize();
  }
  gettimeofday(&t2, NULL);
  double time_alias_table_sample = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
  ave_time_alias_table_sample += time_alias_table_sample;
  ave_time_alias_table_sample=ave_time_alias_table_sample/1000;
  
  printf("Alias Table Sample: %f ms\n", ave_time_alias_table_sample);

  cudaMemcpy(sample_output, d_sample_output, sizeof(int) * (sample_offset_sum), cudaMemcpyDeviceToHost);
  // printf("采样结果:\n");
  // for(int i=0;i<sample_offset_sum;i++) printf("%d, ", sample_output[i]);

FILE *fout = fopen("node2vec-313.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%i,%i,%i,%f\n",
            filename,  matrixA_d->m, matrixA_d->n, matrixA_d->nnz,ave_time_alias_table_sample);
    fclose(fout);
//-------------------------------l2 end--------------------------------

  // FILE *fout = fopen("40-20%-results-4070.csv", "a");
  //   if (fout == NULL)
  //       printf("Writing results fails.\n");
  //   fprintf(fout, "%s,%i,%i,%i,%i,%i,%f,%f\n",
  //           filename,  matrixA_d->m, matrixA_d->n, matrixA_d->nnz,max_rowid,max_rowlen, time_alias_table_constrcut, ave_time_alias_table_sample);
  //   fclose(fout);

  
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
    
 matrixA_d->nnz, matrixA_d->time_cuda_step);
    fclose(fout);
    */
