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
#include <algorithm>
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


int main(int argc, char **argv)
{
  char *filename;
  filename = argv[1];
  char *group;
  group = argv[2];

  printf("MAT: -------------- %s --------------\n", filename);
  Matrix *matrixA_d = (Matrix *)malloc(sizeof(Matrix));

  // load mtx A data to the csr format
  mmio_allinone(&matrixA_d->m, &matrixA_d->n, &matrixA_d->nnz, &matrixA_d->isSymmetric, &matrixA_d->rowpointer, &matrixA_d->columnidx, &matrixA_d->value, filename);

  for (int i = 0; i < matrixA_d->nnz; i++)
    matrixA_d->value[i] = i % 10 + 1;

  int *a=(int*)malloc(matrixA_d->m*sizeof(int));
  for(int i=0;i<matrixA_d->m;i++) {
    a[i]=matrixA_d->rowpointer[i+1]-matrixA_d->rowpointer[i];
  }

  std::sort(a,a+matrixA_d->m,std::great<int>());

  long long cur=0;
  for(int i=0;i<matrixA_d->m;i++) {
    cur+=a[i];
    if(cur*2>=matrixA_d->nnz) {
      if(i*100<=matrixA_d->m) {
        FILE *fout = fopen("Test_matrix.csv", "a");
        if (fout == NULL)
            printf("Writing results fails.\n");
        fprintf(fout, "%s,%s,%i,%i,%i\n",
                group, filename,  matrixA_d->m, matrixA_d->n, matrixA_d->nnz);
        fclose(fout);
      }
      return 0;
    }
  }

  return 0;
}
/* FILE *fout = fopen("results.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%i,%i,%i,%f\n",
            filename,  matrixA_d->m, matrixA_d->n, matrixA_d->nnz, matrixA_d->time_cuda_step);
    fclose(fout);
    */
