__global__
void cuda_step1_kernel(int rowA, int colA, int *d_rowpointerA, int *d_columnindexA, 
    int tilemA, int tilenA, int *d_tile_ptr_A, int *d_flag_t, int row_start_idx, int row_end_idx,int num_threads,int n_tile)
{
   // if(threadIdx.x==1)
    //printf("!!");

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id<num_threads)
    {
//printf("global_id=%d\n",global_id);
    int row=row_start_idx;
    for(int i=row_start_idx;i<=row_end_idx;i++)
    {
        if(d_rowpointerA[row_start_idx]+global_id<d_rowpointerA[i+1])
        {
            row=i;
            break;
        }
    }
    int col=d_columnindexA[d_rowpointerA[row_start_idx]+global_id];
    
    if(atomicAdd(&d_flag_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE],1)==0)
    {
        //d_flag_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE]=1;
        atomicAdd(&d_tile_ptr_A[row/BLOCK_SIZE],1);
    }
   // d_flag_t[row%BLOCK_SIZE*tilenA+col/BLOCK_SIZE]=1;
    //int flag=atomicAdd(&d_flag_t[row%BLOCK_SIZE*tilenA+col/BLOCK_SIZE],1);
   // if()
    //atomicAdd(&d_tile_ptr_A[row/tilemA],1);

}
    __syncthreads();

}
//cuda_step2_kernel<<< num_blocks, 64 >>>(rowA, colA, d_rowpointerA, d_columnindexA, tilemA, tilenA, num_threads, row_start_idx, row_end_idx, d_tile_ptr_A,d_tile_columnidx,d_tile_nnz,d_tile_csr_ptr, numtileA,d_j_col,n_tile, d_flag_t);
__global__
void cuda_step2_kernel(int rowA, int colA, int *d_rowpointerA, int *d_columnindexA, 
                  int tilemA, int tilenA,int num_threads,int row_start_idx, int row_end_idx, MAT_PTR_TYPE *d_tile_ptr_A, int *d_tile_columnidx, 
                  int *d_tile_nnz, int *d_tile_csr_ptr, int numtileA,int *d_j_col,int n_tile,int *d_flag_t, int *d_j_num_t)
{

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id<num_threads)
    {
//printf("global_id=%d\n",global_id);
    int row=row_start_idx;
    for(int i=row_start_idx;i<=row_end_idx;i++)
    {
        if(d_rowpointerA[row_start_idx]+global_id<d_rowpointerA[i+1])
        {
            row=i;
            break;
        }
    }
    int col=d_columnindexA[d_rowpointerA[row_start_idx]+global_id];
    int tileptr=d_tile_ptr_A[row/BLOCK_SIZE];
//printf("row=%d col=%d  tileptr=%d\n",row,col,d_tile_ptr_A[row/BLOCK_SIZE]);
    //int k=0;
    //k=atomicAdd(&d_flag_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE],1);
   //  if(atomicAdd(&d_flag_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE],1)==-1)
    int j=0;
    if(atomicAdd(&d_flag_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE],1)==0)
    {
       // j=atomicAdd(&d_j_col[(row/BLOCK_SIZE)%n_tile],1);
        //d_flag_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE]=j;
        d_j_num_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE]=atomicAdd(&d_j_col[(row/BLOCK_SIZE)%n_tile],1);
        j=d_j_num_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE];
        d_tile_columnidx[tileptr+j]=col/BLOCK_SIZE;
        //printf("d_tile_columnidx=%d  tileptr=%d  j=%d  ptr=%d\n",d_tile_columnidx[tileptr+j],tileptr,j,tileptr+j);
    }
     
//printf("tileptr+j=%d  %d=%d\n",tileptr+j,d_tile_columnidx[tileptr+j],col/BLOCK_SIZE);
//printf("row=%d col=%d  j=%d  tileptr=%d\n",row,col,j,tileptr);
   /* j=d_j_num_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE];
      // __threadfence();
    atomicAdd(&d_tile_nnz[tileptr+j],1);
    atomicAdd(&d_tile_csr_ptr[(tileptr+j)*BLOCK_SIZE+row%BLOCK_SIZE],1);
    __syncthreads();*/

}
__syncthreads();
   
}


__global__
void cuda_step2_2_kernel(int rowA, int colA, int *d_rowpointerA, int *d_columnindexA, 
                  int tilemA, int tilenA,int num_threads,int row_start_idx, int row_end_idx, MAT_PTR_TYPE *d_tile_ptr_A, int *d_tile_columnidx, 
                  int *d_tile_nnz, int *d_tile_csr_ptr, int numtileA,int *d_j_col,int n_tile,int *d_flag_t, int *d_j_num_t)
{

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id<num_threads)
    {
//printf("global_id=%d\n",global_id);
    int row=row_start_idx;
    for(int i=row_start_idx;i<=row_end_idx;i++)
    {
        if(d_rowpointerA[row_start_idx]+global_id<d_rowpointerA[i+1])
        {
            row=i;
            break;
        }
    }
    int col=d_columnindexA[d_rowpointerA[row_start_idx]+global_id];
    int tileptr=d_tile_ptr_A[row/BLOCK_SIZE];
//printf("row=%d col=%d  tileptr=%d\n",row,col,d_tile_ptr_A[row/BLOCK_SIZE]);
    //int k=0;
    //k=atomicAdd(&d_flag_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE],1);
   //  if(atomicAdd(&d_flag_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE],1)==-1)
    int j=d_j_num_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE];
      // __threadfence();
    atomicAdd(&d_tile_nnz[tileptr+j],1);
    atomicAdd(&d_tile_csr_ptr[(tileptr+j)*BLOCK_SIZE+row%BLOCK_SIZE],1);
   // printf("row=%d  tileptr+j=%d  tileptr+j+row%BLOCK_SIZE=%d\n",row,tileptr+j,(tileptr+j)*BLOCK_SIZE+row%BLOCK_SIZE);
  //  __syncthreads();

}
__syncthreads();
   
}


__global__
void cuda_step3_kernel(int rowA, int colA, int *rowpointerA, int *columnindexA, 
                  int tilemA, int tilenA, int numtileA, MAT_PTR_TYPE *tile_ptr_A, int *tile_columnidx, int *tile_nnz, char *Format, 
                  int *tile_csr_ptr, int *blknnz, char *blkwidth, int *hyb_coocount,
                  int * denserowptr, int *densecolptr,
                  int *csr_offset, int *csrptr_offset, int *coo_offset, int *ell_offset, int *hyb_offset, int *dns_offset, int *dnsrow_offset, int *dnscol_offset,int blki_1,int num_tile_row,unsigned char *col_flag,int *new_coocount)
{
    // if(blockIdx.x==1)
  //  printf("blockid----1=%d\n",blockIdx.x);

    if(threadIdx.x==1)
    {
    int begin_blki=blki_1;
    int blockid=blockIdx.x;    
    int blki;
   for(int i=0;i<num_tile_row&&i+begin_blki<tilemA;i++)
    {
        if(tile_ptr_A[begin_blki]+blockid<tile_ptr_A[begin_blki+i+1])
        {//printf("blki=%d  tile_ptr_A[blki]=%d  blockid=%d  tile_ptr_A[blki+i+1]=%d\n",begin_blki,tile_ptr_A[begin_blki],blockid,tile_ptr_A[begin_blki+i+1]);
            blki=i+begin_blki;
            break;
        }
    }
    if(blki==tilemA)
       blki--;
   //  printf("blki=%d\n",blki);
   //if(blockIdx.x==1)
   // printf("blockid----2=%d\n",blockIdx.x);

    
    //printf("step3\n");
       //printf("Format=%d\n",Format[blki]);
        //int tilenum_per_row=tile_ptr_A[blki+1]-tile_ptr_A[blki];
 //printf("blockIdx.x=%d  threadIdx.x=%d  blki=%d  tilenum_per_row=%d\n\n",blockIdx.x,threadIdx.x,blki,tilenum_per_row);
       // 
        int rowlen= blki==tilemA-1 ? rowA-(tilemA-1)*BLOCK_SIZE : BLOCK_SIZE ;
      //  printf("rowlen=%d\n",rowlen);
   //     for (int bi=0;bi<tilenum_per_row;bi++)
   //     {
            int collen = tile_columnidx[tile_ptr_A[begin_blki]+blockid] == tilenA-1 ? colA - (tilenA-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
            int tile_id = tile_ptr_A[begin_blki]+blockid;
            int tilennz = tile_nnz[tile_id +1] - tile_nnz[tile_id];
            int nnzthreshold = rowlen * collen * 0.5 ;
       // printf("tilenum_per_row=%d  rowlen=%d  collen=%d  tile_id=%d  tilennz=%d  nnzthreshold=%d\n",tilenum_per_row,rowlen,collen,tile_id,tilennz,nnzthreshold);
 //printf("tile_id=%d  tilennz=%d  blki=%d\n",tile_id,tilennz,blki);
            // if (1)
            // {
            //             Format[tile_id] =0 ;
            //             blknnz[tile_id] = tilennz ;
            //             csr_offset[tile_id] = tilennz;
            //             csrptr_offset[tile_id] = rowlen;
            // }
          //  printf("tilennz=%d  nnzthreshold=%d  COO_THRESHOLD%d\n",tilennz,nnzthreshold,COO_THRESHOLD);
if(1) 
{
                         Format[tile_id] =0 ;
                        blknnz[tile_id] = tilennz ;
                        csr_offset[tile_id] = tilennz;
                        csrptr_offset[tile_id] = BLOCK_SIZE;
             /*   int bwidth=0;
                //int hybwidth=0;
                for (int blkj=0;blkj<rowlen;blkj++)
                {
                    if (bwidth < tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] )
                        bwidth = tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] ;
                }

                    Format[tile_id] = 2;
                   // printf("2222222222222222222\n");
                    blkwidth[tile_id]=bwidth;
                    blknnz[tile_id] = bwidth * rowlen ;
                    ell_offset[tile_id] = bwidth * rowlen;*/
           /*     Format[tile_id] = 1 ;
                //printf("111111111111  blockid=%d\n",blockid);
                blknnz[tile_id] = tilennz;
                coo_offset[tile_id] = tilennz;
                new_coocount[tile_id] = tilennz;*/
          /*      int bwidth=0;
                //int hybwidth=0;
                for (int blkj=0;blkj<rowlen;blkj++)
                {
                    if (bwidth < tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] )
                        bwidth = tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] ;
                }
                    int hybwidth=bwidth;
int iopriorsize=   bwidth * rowlen * sizeof (MAT_VAL_TYPE) + bwidth * rowlen * sizeof (char)  ;
              //      int iopriorsize=  bwidth * rowlen * sizeof (MAT_VAL_TYPE) + bwidth * rowlen * sizeof (unsigned char)  ;
                                                                // bwidth * rowlen * sizeof (MAT_VAL_TYPE) + bwidth * rowlen * sizeof (char) /2 +1 ;
                    int ionextsize;
                    int coonextnum=0;
                    int coopriornum=0;
                    for (int wi=bwidth-1;wi>0;wi--)
                    {
                        coonextnum=0;
                        for (int blkj=0;blkj<rowlen;blkj++)
                        {
                            if ( tile_csr_ptr[tile_id * BLOCK_SIZE + blkj]> wi) 
                                {
                                    coonextnum += tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] - wi ;
                                } 
                        }
                        ionextsize=  wi * rowlen * sizeof (MAT_VAL_TYPE )+  wi * rowlen * sizeof ( char)  + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (unsigned char)) ;
                                                                // wi * rowlen * sizeof (MAT_VAL_TYPE )+  wi * rowlen * sizeof (char) /2 + 1 + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (char)) ;
                        if (iopriorsize<=ionextsize)
                        {
                            hybwidth=wi+1;
                            break;
                        }
                        else
                        {
                            hybwidth = wi;
                            iopriorsize=ionextsize;
                            coopriornum=coonextnum;
                        }

                    }
                    

                        Format[tile_id] = 3;
                        hyb_coocount[tile_id] = coopriornum;
                        blkwidth[tile_id]=hybwidth;
                        blknnz[tile_id] = coopriornum + hybwidth * rowlen ;
                        hyb_offset[tile_id] = coopriornum + hybwidth * rowlen;
                        new_coocount[tile_id] = coopriornum;*/

             /*   Format[tile_id] = 4 ;
               // printf("Format=%d\n",Format[blki]);
               // printf("444444444444  blockid=%d\n",blockid);
                blknnz[tile_id] = rowlen * collen;
                dns_offset[tile_id] = rowlen * collen;*/
//int rowlen= blki==tilemA-1 ? rowA-(tilemA-1)*BLOCK_SIZE : BLOCK_SIZE ;
//if(blki==tilemA-1)
//printf("blki=%d tilemA-1=%d  rowA=%d  rowlen=%d   rowA-(tilemA-1)*BLOCK_SIZE=%d\n",blki,tilemA-1,rowA,rowlen,rowA-(tilemA-1)*BLOCK_SIZE);
                return;

                    

               
}
else
{
            if (tilennz >= nnzthreshold)  //if the number of nnz is more than 128, then dense
            {  
                Format[tile_id] = 4 ;
               // printf("Format=%d\n",Format[blki]);
               // printf("444444444444  blockid=%d\n",blockid);
                blknnz[tile_id] = rowlen * collen;
                dns_offset[tile_id] = rowlen * collen;
                return;
            }
        if (tilennz <= COO_THRESHOLD) //else if the number of nnz is less than 12, then coo
            {
                Format[tile_id] = 1 ;
                //printf("111111111111  blockid=%d\n",blockid);
                blknnz[tile_id] = tilennz;
                coo_offset[tile_id] = tilennz;
                new_coocount[tile_id] = tilennz;
                return;
            }  
 //if(blockid==1)
   // printf("blockid----1=%d\n",blockid);
   //---------------------------------------------------------------------------------
            else if (tilennz % collen ==0 || tilennz % rowlen ==0)
            {
                int dnsrowflag =0 ;
                int numdnsrow =0;
                int dnscolflag =0;
                int numdnscol =0;
                for (int ri=0;ri < rowlen ;ri++)
                {
                    if (tile_csr_ptr[tile_id * BLOCK_SIZE + ri] % collen !=0)
                    {
                        dnsrowflag =0;
                        break;
                    }
                    else 
                    {
                        if (tile_csr_ptr[tile_id * BLOCK_SIZE + ri]  == collen)
                        {
                            dnsrowflag =1;
                            numdnsrow ++ ;
                        }
                    }
                    
                }
                if (dnsrowflag  == 1)
                {                    
                    Format[tile_id] = 5 ;   //Dense Row
                   // printf("55555555555 blockid=%d\n",blockid);
                    denserowptr[tile_id] = numdnsrow ;
                    blknnz[tile_id] = numdnsrow * collen;
                    dnsrow_offset[tile_id] = numdnsrow * collen;
                    return;
                }
                else 
                {
                    int start = blki*BLOCK_SIZE;
                    int end = blki==tilemA-1 ?  rowA : (blki+1)*BLOCK_SIZE ;
                    int jc = tile_columnidx[tile_id];
                    //printf("tile_id=%d  start=%d  end=%d  rowpointerA[start]=%d  rowpointerA[end]=%d\n",tile_id,start,end,rowpointerA[start],rowpointerA[end]);
                  //  printf("start=%d  end=%d\n",start,end);
                    //printf("start=%d  end=%d   rowpointerA[start]=%d  rowpointerA[end]=%d\n",start,end,rowpointerA[start],rowpointerA[end]);
                  //  printf("ggggggggggggggggg");
                   // unsigned char *dnscol_colidx_temp= (unsigned char *)malloc(tilennz * sizeof(unsigned char));
                  //  memset(dnscol_colidx_temp, -1, tilennz * sizeof(unsigned char));
                  //  int k=0;
                    //unsigned char *col_flag =(unsigned char *)malloc(collen * sizeof(unsigned char));
                  
                   // memset(col_flag, 0, collen * sizeof(unsigned char));
                  //  __shared__  unsigned char col_flag[17]={0};
                   for (int blkj = rowpointerA[start]; blkj < rowpointerA[end]; blkj ++)
                    {
                        int jc_temp = columnindexA[blkj]/BLOCK_SIZE;
   //printf("tile_id=%d  blockid=%d  blkj=%d  columnindexA[blkj]=%d  jc=%d  jc_temp=%d\n",tile_id,blockid,blkj,columnindexA[blkj],jc,jc_temp);
                      if (jc_temp == jc)
                        {
                            int col_temp = columnindexA[blkj] - jc * BLOCK_SIZE;
                            int k=blockid*BLOCK_SIZE+col_temp;
                            //printf("ggggggggggggggggg");
                            //printf("k=%d\n",k);
                            col_flag[blockid*BLOCK_SIZE+col_temp] ++;
                            //printf("tile_id=%d  k=%d  blockid=%d  col_temp=%d  col_flag[blockid*BLOCK_SIZE+col_temp]=%d  blkj=%d  columnindexA[blkj]=%d  jc=%d\n",tile_id,k,blockid,col_temp,col_flag[blockid*BLOCK_SIZE+col_temp],blkj,columnindexA[blkj],jc);
                        }

                    }
                     for (int j =0; j < collen; j ++)
                    {
                        if (col_flag[blockid*BLOCK_SIZE+j] % rowlen !=0)
                        {
                            dnscolflag =0;
                            break;
                        }
                        else 
                        {
                           if (col_flag[blockid*BLOCK_SIZE+j] == rowlen)
                            {
                                dnscolflag =1;
                                numdnscol ++ ;
                            }
                        }
                    }
                    if (dnscolflag  == 1)
                    {                    
                //        printf("numdnscol = %i\n", numdnscol);
                        Format[tile_id] = 6 ;   //Dense Col
                        //printf("666666666666 blockid=%d\n",blockid);
                        densecolptr[tile_id] = numdnscol ;
                        blknnz[tile_id] = numdnscol * rowlen;
                        dnscol_offset[tile_id] = numdnscol * rowlen;
                        return;
                    }            

                
                }
                //printf("Format=%d\n",Format[blki]);
            }
            //printf("rowlen=%d\n",rowlen);
            if (Format[tile_id] != 5 && Format[tile_id] !=6)
            {
                int bwidth=0;
                //int hybwidth=0;
                for (int blkj=0;blkj<rowlen;blkj++)
                {
                    if (bwidth < tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] )
                        bwidth = tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] ;
                }
                double tilennz_d=(double)tilennz;
                double row_length_mean = (double)(tilennz_d / rowlen);
                double variance             = 0.0;
                double row_length_skewness   = 0.0;
                //printf("row_length_mean=%f  tilennz=%d  tilennz_d=%f rowlen=%d\n",row_length_mean,tilennz,tilennz_d,rowlen);
               // printf("rowlen=%d\n",rowlen);
                double rowlen_d=rowlen;
                for (int row = 0; row < rowlen; ++row)
                {
                    int length              = tile_csr_ptr[tile_id * BLOCK_SIZE + row];
                    double delta                = (double)((double)length - row_length_mean);
                    
//printf("delta=%f\n",delta);
                    variance   += (delta * delta);
                   // printf("cuda: tile_id=%d  row=%d  length=%d  row_length_mean =%f delta=%f  variance=%f\n",tile_id,row,length,row_length_mean,delta,variance);
                    row_length_skewness   += (delta * delta * delta);
        //  printf("row=%d\n",row);
                }
                //printf("variance=%f   rowlen=%d  rowlen_d=%f\n",variance,rowlen,rowlen_d);
                variance                    /= rowlen;
                //printf("variance=%f\n",variance);
                double row_length_std_dev    = sqrt(variance);
                row_length_skewness   = (row_length_skewness / rowlen) / pow(row_length_std_dev, 3.0);
                double row_length_variation  = row_length_std_dev / row_length_mean;

                double ell_csr_threshold = 0.2;
                double csr_hyb_threshold = 1.0;
             //   printf("row_length_variation=%f  row_length_std_dev=%f  row_length_mean=%f  variance=%f  tilennz=%d  rowlen=%d \n",row_length_variation,row_length_std_dev,row_length_mean,variance,tilennz,rowlen);
                if (row_length_variation <= ell_csr_threshold)  // if variation is less than 0.2, then ELL
                {
                    Format[tile_id] = 2;
                   // printf("2222222222222222222\n");
                    blkwidth[tile_id]=bwidth;
                    blknnz[tile_id] = bwidth * rowlen ;
                    ell_offset[tile_id] = bwidth * rowlen;
                }
              /*  else
                {
                    int hybwidth=bwidth;
                    int iopriorsize =   bwidth * rowlen * sizeof (MAT_VAL_TYPE) + bwidth * rowlen * sizeof (char)  ;
                                                                // bwidth * rowlen * sizeof (MAT_VAL_TYPE) + bwidth * rowlen * sizeof (char) /2 +1 ;
                    int ionextsize;
                    int coonextnum=0;
                    int coopriornum=0;
                    for (int wi=bwidth-1;wi>=0;wi--)
                    {
                        coonextnum=0;
                        for (int blkj=0;blkj<rowlen;blkj++)
                        {
                        if ( tile_csr_ptr[tile_id * BLOCK_SIZE + blkj]> wi) 
                            {
                                coonextnum += tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] - wi ;
                            } 
                        }
                        ionextsize=  wi * rowlen * sizeof (MAT_VAL_TYPE )+  wi * rowlen * sizeof ( char)  + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (unsigned char)) ;
                                                                // wi * rowlen * sizeof (MAT_VAL_TYPE )+  wi * rowlen * sizeof (char) /2 + 1 + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (char)) ;
                    
                        if (iopriorsize<=ionextsize)
                        {
                            hybwidth=wi+1;
                            break;
                        }
                        else
                        {
                            hybwidth = wi;
                            iopriorsize=ionextsize;
                            coopriornum=coonextnum;
                        }

                    }
                    if (row_length_variation >= csr_hyb_threshold )//&& coopriornum <= 4)  // if variation > 1.0, and the number of coo data <=4, then HYB
                    {
                        Format[tile_id] = 3;
                     //   printf("33333333333333333333\n");
                        hyb_coocount[tile_id] = coopriornum;
                        blkwidth[tile_id]=hybwidth;
                        blknnz[tile_id] = coopriornum + hybwidth * rowlen ;
                        hyb_offset[tile_id] = coopriornum + hybwidth * rowlen;
                        new_coocount[tile_id] = coopriornum;
//printf("3 tile_id=%d: row_length_variation=%f   variance=%f  tilennz=%d  rowlen=%d \n",tile_id,row_length_variation,variance,tilennz,rowlen);
//printf("hyb_coocount[tile_id]=%d\n",hyb_coocount[tile_id]);
                    }
                    else  //else CSR
                    {
                        Format[tile_id] =0 ;
                      //  printf("000000000000000000000\n");
                        blknnz[tile_id] = tilennz ;
                        csr_offset[tile_id] = tilennz;
                        csrptr_offset[tile_id] = BLOCK_SIZE;
//printf("tile_id=%d: csr_offset[tile_id]=%d\n",tile_id,csr_offset[tile_id]);
//printf("0 tile_id=%d: row_length_variation=%f   variance=%f  tilennz=%d  rowlen=%d \n",tile_id,row_length_variation,variance,tilennz,rowlen);
                    }
                    
                }*/
                else
                {
                    int hybwidth=bwidth;
int iopriorsize=   bwidth * rowlen * sizeof (MAT_VAL_TYPE) + bwidth * rowlen * sizeof (char)  ;
              //      int iopriorsize=  bwidth * rowlen * sizeof (MAT_VAL_TYPE) + bwidth * rowlen * sizeof (unsigned char)  ;
                                                                // bwidth * rowlen * sizeof (MAT_VAL_TYPE) + bwidth * rowlen * sizeof (char) /2 +1 ;
                    int ionextsize;
                    int coonextnum=0;
                    int coopriornum=0;
                    for (int wi=bwidth-1;wi>0;wi--)
                    {
                        coonextnum=0;
                        for (int blkj=0;blkj<rowlen;blkj++)
                        {
                            if ( tile_csr_ptr[tile_id * BLOCK_SIZE + blkj]> wi) 
                                {
                                    coonextnum += tile_csr_ptr[tile_id * BLOCK_SIZE + blkj] - wi ;
                                } 
                        }
                        ionextsize=  wi * rowlen * sizeof (MAT_VAL_TYPE )+  wi * rowlen * sizeof ( char)  + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (unsigned char)) ;
                                                                // wi * rowlen * sizeof (MAT_VAL_TYPE )+  wi * rowlen * sizeof (char) /2 + 1 + coonextnum * (sizeof (MAT_VAL_TYPE) + sizeof (char)) ;
                        if (iopriorsize<=ionextsize)
                        {
                            hybwidth=wi+1;
                            break;
                        }
                        else
                        {
                            hybwidth = wi;
                            iopriorsize=ionextsize;
                            coopriornum=coonextnum;
                        }

                    }
                    
                    if (row_length_variation >= csr_hyb_threshold )//&& coopriornum <= 4)  // if variation > 1.0, and the number of coo data <=4, then HYB
                    {
                        Format[tile_id] = 3;
                        hyb_coocount[tile_id] = coopriornum;
                        blkwidth[tile_id]=hybwidth;
                        blknnz[tile_id] = coopriornum + hybwidth * rowlen ;
                        hyb_offset[tile_id] = coopriornum + hybwidth * rowlen;
                        new_coocount[tile_id] = coopriornum;

                    }
                    else  //else CSR
                    {
                        Format[tile_id] =0 ;
                        blknnz[tile_id] = tilennz ;
                        csr_offset[tile_id] = tilennz;
                        csrptr_offset[tile_id] = BLOCK_SIZE;
                    }
                    
                }
             //printf("Format=%d\n",Format[blki]);
           //printf("Format[tile_id]=%d\n",Format[tile_id]); 
            //}
        }
    }
//printf("blockid=%d\n",blockid);
    }
__syncthreads();
}


__inline__ __device__
void exclusive_scan_cu(MAT_PTR_TYPE *input, int length)
{
    if(length == 0 || length == 1)
        return;
    
    MAT_PTR_TYPE old_val, new_val;
    
    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}

__global__
void cuda_step4_kernel_1(int rowA, int colA, int *rowpointerA, int *columnindexA, MAT_VAL_TYPE *valueA,
                  int tilemA, int tilenA, int numtileA, MAT_PTR_TYPE *tile_ptr_A, int *tile_columnidx, int *tile_nnz, char *Format, 
                  int *blknnz, int *csr_ptr,  int nnz_temp,  int tile_count_temp,
                  unsigned char  *csr_colidx_temp_g,MAT_VAL_TYPE *csr_val_temp_g,int *tile_count_g,int blki_1,int num_tile_row)
{

    //for each tile
  //  #pragma omp parallel for  
   if(threadIdx.x==1)
    {
        int begin_blki=blki_1;
        int blockid=blockIdx.x;    
        int blki=begin_blki+blockid;
   /*      for(int i=0;i<num_tile_row&&i+begin_blki<tilemA;i++)
        {
            if(tile_ptr_A[begin_blki]+blockid<tile_ptr_A[begin_blki+i+1])
            {printf("blki=%d  tile_ptr_A[blki]=%d  blockid=%d  tile_ptr_A[blki+i+1]=%d\n",begin_blki,tile_ptr_A[begin_blki],blockid,tile_ptr_A[begin_blki+i+1]);
                blki=i+begin_blki;
                break;
            }
        }
        if(blki==tilemA)
            blki--;*/
  //printf("blki=%d\n",blki);
        int thread_id = blockIdx.x;
        unsigned char  *csr_colidx_temp = csr_colidx_temp_g + thread_id * nnz_temp;
        MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + thread_id * nnz_temp;
        int *tile_count = tile_count_g + thread_id * tile_count_temp;
      //  printf("blki=%d  thread_id=%d  thread_id * nnz_temp=%d\n",blki,thread_id,thread_id * nnz_temp);
     //   printf("thread_id=%d  \n",thread_id);
        // unsigned char  *csr_colidx_temp = (unsigned char *)malloc((nnz_temp )*sizeof(unsigned char));
        // MAT_VAL_TYPE *csr_val_temp = (MAT_VAL_TYPE *)malloc((nnz_temp)*sizeof(MAT_VAL_TYPE));
        // int *tile_count = (int *)malloc(tile_count_temp * sizeof(int));
        
        //memset(csr_colidx_temp, 0, (nnz_temp)*sizeof(unsigned char));
        //memset(csr_val_temp, 0, (nnz_temp)*sizeof(MAT_VAL_TYPE));
        //memset(tile_count, 0, (tile_count_temp)*sizeof(int));
        int tilenum_per_row=tile_ptr_A[blki+1]-tile_ptr_A[blki];
        int rowlen= blki==tilemA-1 ? rowA-(tilemA-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int start = blki*BLOCK_SIZE;
        int end = blki==tilemA-1 ?  rowA : (blki+1)*BLOCK_SIZE ;
        //printf("")
        // if (blki == 978)
        // {
        //     printf("thread_id= ,tilenum_per_row=%i, nnz = %i\n", tilenum_per_row, rowpointerA[end]-rowpointerA[start]);
        //     printf("start = %i, end = %i\n",start, end);
        // }
        //printf("start=%d  end=%d  tilenum_per_row=%d\n",start,end,tilenum_per_row);
       for (int blkj = rowpointerA[start]; blkj < rowpointerA[end]; blkj ++)
        {
           int jc_temp = columnindexA[blkj]/BLOCK_SIZE;
            //printf("blkj = %i,col=%i\n", blkj, jc_temp);
             for (int bi = 0; bi < tilenum_per_row; bi ++)
            {
                int tile_id = tile_ptr_A[blki]+bi;
                int jc = tile_columnidx[tile_id];
                int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr_A[blki]];
               
                 if (jc == jc_temp)
                {//printf("tile_id = %d  blkj=%d  jc=%d  pre_nnz=%d\n", tile_id, blkj, jc, pre_nnz);
                  /*  if(blkj>59511){
                   printf("tile_id = %d, jc=%d  pre_nnz=%d\n", tile_id, jc, pre_nnz);}*/
                    csr_val_temp[pre_nnz + tile_count[bi]] = valueA[blkj];
                    csr_colidx_temp[pre_nnz + tile_count[bi]] = columnindexA[blkj] - jc * BLOCK_SIZE;
                    //printf("tile_id=%d  jc=%d  blkj=%d  pre_nnz + tile_count[bi]=%d csr_val_temp=[]=%d \n",tile_id,jc,blkj,pre_nnz + tile_count[bi],csr_val_temp[pre_nnz + tile_count[bi]]);
                    // printf("tile_id=%d  k1=%d  tilennz=%d  offset + k1=%d  Tile_csr_Val[offset + k1]=%f  pre_nnz=%d\n",tile_id, k1,tilennz,offset + k1,Tile_csr_Val[offset + k1],pre_nnz);
                //       printf("tile_id = %i, tilennz = %i, jc = %i, prennz = %i, val[%i]=%f,col_before= %i, col[] = %i\n",tile_id, tilennz, jc, pre_nnz,pre_nnz + tile_count[bi],csr_val_temp[pre_nnz + tile_count[bi]], columnindexA[blkj],csr_colidx_temp[pre_nnz + tile_count[bi]]);//
                    //printf("tile_id = %i,  jc = %i, prennz = %i, val[%i]=%f,col_before= %i, col[] = %i  bi=%d  tile_count[bi]=%d\n",tile_id,  jc, pre_nnz,pre_nnz + tile_count[bi],csr_val_temp[pre_nnz + tile_count[bi]], columnindexA[blkj],csr_colidx_temp[pre_nnz + tile_count[bi]],bi,tile_count[bi]);
                    tile_count[bi] ++;  //woshicuode
                    break;
                      
                }
            }
        }
    }
__syncthreads();
}




__global__
void cuda_step4_kernel_2(int rowA, int colA, int *rowpointerA, int *columnindexA, MAT_VAL_TYPE *valueA,
                  int tilemA, int tilenA, int numtileA, MAT_PTR_TYPE *tile_ptr_A, int *tile_columnidx, int *tile_nnz, char *Format, 
                  int *blknnz, int *csr_ptr,  int nnz_temp,  int tile_count_temp,
                  unsigned char  *csr_colidx_temp_g,MAT_VAL_TYPE *csr_val_temp_g,int *tile_count_g,
                 MAT_VAL_TYPE *Tile_csr_Val, unsigned char  *Tile_csr_Col, unsigned char  *Tile_csr_Ptr, int *csr_offset, int *csrptr_offset,
                 MAT_VAL_TYPE *Tile_coo_Val, unsigned char  *Tile_coo_colIdx, unsigned char  *Tile_coo_rowIdx, int *coo_offset,
                 MAT_VAL_TYPE *Tile_ell_Val, unsigned char  *Tile_ell_colIdx, char *blkwidth, int *ell_offset,
                 MAT_VAL_TYPE *Tile_hyb_Val, unsigned char  *Tile_hyb_ellcolIdx, unsigned char  *Tile_hyb_coorowIdx, int * hyb_coocount, int *hyb_offset,
                 MAT_VAL_TYPE *Tile_dns_Val, int *dns_offset,
                 MAT_VAL_TYPE *Tile_denserow_Val, char *Tile_dnsrow_idx, int *denserowptr, int *dnsrow_offset,
                 MAT_VAL_TYPE *Tile_dnscol_Val, char *Tile_dnscol_idx,  int *densecolptr, int *dnscol_offset,
                 int blki_1, int num_tile_row,int *new_coocount, MAT_VAL_TYPE *new_coo_value, int *new_coo_colidx, int *new_coo_rowidx)
{
    if(threadIdx.x==1)
    {//printf("hhh\n");
        int begin_blki=blki_1;
        int blockid=blockIdx.x;    
        int blki;
        for(int i=0;i<num_tile_row&&i+begin_blki<tilemA;i++)
        {
            if(tile_ptr_A[begin_blki]+blockid<tile_ptr_A[begin_blki+i+1])
            {//printf("blki=%d  tile_ptr_A[blki]=%d  blockid=%d  tile_ptr_A[blki+i+1]=%d\n",begin_blki,tile_ptr_A[begin_blki],blockid,tile_ptr_A[begin_blki+i+1]);
                blki=i+begin_blki;
                break;
            }
        }
        if(blki==tilemA)
            blki--;
        //printf("blki=%d\n",blki);
        //printf("blockid=%d\n",blockid);
        MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + (blki%num_tile_row) * nnz_temp;
        unsigned char  *csr_colidx_temp = csr_colidx_temp_g + (blki%num_tile_row) * nnz_temp;
        
        int rowlen= blki==tilemA-1 ? rowA-(tilemA-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int bi = blockIdx.x;
            int tile_id = tile_ptr_A[begin_blki]+bi;
//printf("blki=%d  tile_id=%d   (blkinum_tile_row) * nnz_temp=%d\n",blki,tile_id,(blki%num_tile_row) * nnz_temp);
            int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr_A[blki]];
            int tilennz = tile_nnz[tile_id +1] - tile_nnz[tile_id];
            int collen = tile_columnidx[tile_id] == tilenA-1 ? colA - (tilenA-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
            int format = Format[tile_id];
            switch (format)
            {
                case 0:
                {
                    int offset = csr_offset[tile_id];
                    int ptr_offset = csrptr_offset[tile_id];
                 //   printf("tile_id=%d  csr_offset[tile_id]=%d \n",tile_id,csr_offset[tile_id]);
                    int *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_cu(ptr_temp, rowlen);
                    int k1=0;
 // printf("tile_id=%d  k1=%d    \n",tile_id,k1);
//printf("hhhhhh-tile_id=%d  k1=%d   tilennz=%d \n",tile_id,k1, tilennz);

                    for (; k1 < tilennz; k1++)
                    {
                        Tile_csr_Val[offset + k1] = csr_val_temp[pre_nnz + k1];
                        Tile_csr_Col[offset + k1] = csr_colidx_temp[pre_nnz + k1];
                       // printf("tile_id=%d  offset=%d  k1=%d  tilennz=%d  offset + k1=%d  Tile_csr_Val[offset + k1]=%f  pre_nnz=%d\n",tile_id,offset, k1,tilennz,offset + k1,Tile_csr_Val[offset + k1],pre_nnz);
                      // printf("tile_id=%d  k1=%d   tilennz=%d \n",tile_id,k1, tilennz);

//printf("h");
                    }
                    //CSR ptr
                    for (int pid=0; pid<rowlen; pid++)
                    {
                        Tile_csr_Ptr[ptr_offset+ pid] =ptr_temp[pid] ;
                        //printf("tile_id=%d  blockid=%d  ptr_offset+ pid:=%d Tile_csr_Ptr[ptr_offset+ pid]=%d\n",tile_id,blockid,ptr_offset+ pid,Tile_csr_Ptr[ptr_offset+ pid]);
                       // printf("tile_id=%d  tile_csr_ptr = %i  ptr_offset=%d  pid=%d  ptr_offset+ pid=%d , csr_ptr = %i\n", tile_id, Tile_csr_Ptr[ptr_offset+ pid] ,ptr_offset, pid,ptr_offset+ pid ,csr_ptr[tile_id * BLOCK_SIZE + pid]);

                    }
                    // unsigned char old_val = Tile_csr_Ptr[ptr_offset];
                    // unsigned char new_val;
                    // Tile_csr_Ptr[ptr_offset] =0;
                    // for (int pid =1; pid < BLOCK_SIZE; pid ++)
                    // {
                    //     new_val = Tile_csr_Ptr[ptr_offset+pid];
                    //     Tile_csr_Ptr[ptr_offset+pid] = old_val + Tile_csr_Ptr[ptr_offset+pid -1];
                    //     old_val = new_val;
                    // }
                    break;
                }

                case 1:
                {
                    int colidx_temp = tile_columnidx[tile_id];

                    int offset_new = new_coocount[tile_id];
                    int *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_cu(ptr_temp, BLOCK_SIZE);
//int offset_new=0;
                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ ri +1];
                        for (int j = ptr_temp[ri]; j < nnz_end; j++)
                        {//printf("tile_id=%d offset_new=%d j=%d offset_new + j=%d\n",tile_id,offset_new , j,offset_new + j);
                            new_coo_rowidx[offset_new + j] = ri + blki * BLOCK_SIZE;
                            new_coo_value[offset_new + j] = csr_val_temp[pre_nnz + j] ;
                            new_coo_colidx[offset_new + j]=csr_colidx_temp[pre_nnz + j] + colidx_temp * BLOCK_SIZE;

                        }
                    }
                   /* int offset = coo_offset[tile_id];

                    int *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_cu(ptr_temp, rowlen);


                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ ri +1];
                        for (int j = ptr_temp[ ri]; j < nnz_end; j++)
                        {
                            Tile_coo_rowIdx[offset+ j] = ri;
                            Tile_coo_Val[offset + j] = csr_val_temp[pre_nnz + j] ;
                            Tile_coo_colIdx[offset + j]=csr_colidx_temp[pre_nnz + j];
                        }
                    }*/
                    
                    break;
                 }

                case 2:
                {
                    int offset = ell_offset[tile_id];
                    int *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_cu(ptr_temp, rowlen);
                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ri +1];

                        for (int j = ptr_temp[ri]; j < nnz_end; j++)
                        {
                            int temp = j - ptr_temp[ri];
                            Tile_ell_colIdx[offset + temp * rowlen + ri] = csr_colidx_temp[pre_nnz + j];
                            Tile_ell_Val[offset + temp * rowlen + ri] = csr_val_temp[pre_nnz + j];
                        }
                    }
                    
                    break;
                }
                case 3:
                {
                    int colidx_temp = tile_columnidx[tile_id];
                    int offset = hyb_offset[tile_id];
                    int *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_cu(ptr_temp, BLOCK_SIZE);
                    int offset_new = new_coocount[tile_id];
                    int coocount=0;
                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ri +1];
                        int stop= (nnz_end- ptr_temp[ri]) <= blkwidth[tile_id] ? nnz_end : ptr_temp[ri] + blkwidth[tile_id] ;
                            
                        for (int j = ptr_temp[ri]; j < stop; j++)
                        {
                            int temp = j - ptr_temp[ri];
                            Tile_hyb_ellcolIdx[offset + temp * rowlen + ri] = csr_colidx_temp[pre_nnz + j];
                            Tile_hyb_Val[offset + temp * rowlen + ri] = csr_val_temp[pre_nnz + j];
                        }
                      /* for (int k=stop; k< nnz_end; k++)
                        {
                            Tile_hyb_Val[offset + blkwidth[tile_id] * rowlen + coocount] = csr_val_temp[pre_nnz +k];
                            Tile_hyb_ellcolIdx[offset + blkwidth[tile_id] * rowlen + coocount] = csr_colidx_temp[pre_nnz +k];
                            Tile_hyb_coorowIdx[hyb_coocount[tile_id] + coocount] = ri;
                            coocount++;  
                        }*/
                        for (int k=stop; k< nnz_end; k++)
                        {
                     	   new_coo_value[offset_new + coocount] = csr_val_temp[pre_nnz +k];
                      	   new_coo_colidx[offset_new+coocount] = csr_colidx_temp[pre_nnz +k] + colidx_temp * BLOCK_SIZE;
                      	   new_coo_rowidx[offset_new+coocount] = ri + blki * BLOCK_SIZE;
                      	   coocount++;  
                    	}
                    }
                    
                    break;
                    
                }
                    
                case 4:
                {
                    int offset = dns_offset[tile_id];
                    int *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_cu(ptr_temp, rowlen);

                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ri +1];

                        for (int j = ptr_temp[ri]; j < nnz_end; j++)
                        {
                            Tile_dns_Val[offset + csr_colidx_temp[pre_nnz + j] * rowlen +ri] = csr_val_temp[pre_nnz + j];
                        //    Blockdense_Val[dnsnum[rowblock_ptr[rbi]+bi] + subrowmatrixA[bi].columnindex[j] * rowlength + ri]= subrowmatrixA[bi].value[j];
                        }
                    }
                    
                    break;
                }
                
                case 5:
                {
                    int offset = dnsrow_offset[tile_id];
                    int rowoffset = denserowptr[tile_id];
                    int *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_cu(ptr_temp, rowlen);

                    int dnsriid=0;
                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ri +1];
                        if (nnz_end - ptr_temp[ri] == collen)
                        {
                            // printf("tileid = %i, offset = %i, rowoffset = %i, num = %i\n", tile_id, offset, rowoffset, csr_ptr[tile_id * BLOCK_SIZE + ri]);
                            Tile_dnsrow_idx[rowoffset + dnsriid]=ri;
                            dnsriid ++;
                            for (int j = ptr_temp[ri]; j < nnz_end; j++)
                            {
                                Tile_denserow_Val[offset + j] = csr_val_temp[pre_nnz + j];
                            }
                        }
                    }
                    break;
                }
                
                  

                case 6:
                {
                    int offset = dnscol_offset[tile_id];
                    int coloffset = densecolptr[tile_id];
                    int *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                    exclusive_scan_cu(ptr_temp, rowlen);

                    // for (int ni =0; ni < rowlen  ; ni ++)
                    // {
                    //     printf("%i    ", ptr_temp[ni]);
                    // }
                    // printf("\n");

                    int dnsciid=0;
                    for (int j=ptr_temp[0];j < ptr_temp[1];j ++)
                    {
                        int ci = csr_colidx_temp[pre_nnz + j];
                        // int ci = subrowmatrixA[bi].columnindex[j] ;
                        Tile_dnscol_idx[coloffset + dnsciid] =ci ;
                //        printf("pos=%i, colidx=%i\n",densecolptr[tile_id + dnsciid],Tile_dnscol_idx[coloffset + dnsciid] );
                        dnsciid++;
                    }
                    for (int ri = 0; ri < rowlen; ri++)
                    {
                        int nnz_end = ri == rowlen -1 ? tilennz : ptr_temp[ri +1];

                        for (int j = ptr_temp[ri]; j < nnz_end; j++)
                        {
                            int temp = j - ptr_temp[ri];
                            if (csr_val_temp[pre_nnz +j] != 0)
                        //    printf("idx = %i, col=%i, val = %f\n",pre_nnz +j, csr_colidx_temp[pre_nnz +j] , csr_val_temp[pre_nnz +j]);
                            Tile_dnscol_Val[offset + temp * rowlen + ri] = csr_val_temp[pre_nnz +j];
                        }
                    }
                    break;
                 }
                
                default:
                    break;
            }

        }
__syncthreads();

    }



__global__
void cuda_coo_rowptrnum_kernel(int nnz_1,int rowA,int colA,int *new_nnz_count, int start, int end,int *new_coo_rowidx,int num_threads)
{
   // if(threadIdx.x==1)
    //printf("!!");

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id<num_threads)
    {
//printf("global_id=%d\n",global_id);
        int row=new_coo_rowidx[start+global_id];
        atomicAdd(&new_nnz_count[row],1);
    }
__syncthreads();

}

      /*  int row = new_coo_rowidx[i];
        int nnz_offset = new_nnz_count[row];
        matrixA->coo_new_value[nnz_offset + new_num[ row]] = new_coo_value[i];
        matrixA->coo_new_colidx[nnz_offset + new_num[row]] = new_coo_colidx[i];
        new_num[row] ++;*/
//cuda_coo_kernel<<<num_blocks, 32 >>>(nnz_1,rowA, colA,d_new_nnz_count,start,end,d_new_coo_rowidx,num_threads,
 //                d_coo_new_colidx,d_coo_new_value,d_new_num,d_new_coo_value_1,d_new_coo_colidx);
__global__
void cuda_coo_kernel(int nnz_1,int rowA,int colA,int *new_nnz_count, int start, int end,int *new_coo_rowidx,int num_threads, 
                     int *coo_new_colidx, MAT_VAL_TYPE *coo_new_value, int *new_num, MAT_VAL_TYPE *new_coo_value,int *new_coo_colidx)
{
   // if(threadIdx.x==1)
    //printf("!!");

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id<num_threads)
    {   int i=start+global_id;
        int row=new_coo_rowidx[i];
        int nnz_offset = new_nnz_count[row];
//printf("i=%d  global_id=%d  row=%d  nnz_offset=%d\n",i,global_id,row,nnz_offset);
        int x=new_num[row];

        int tmp=atomicAdd(&new_num[row],1);
        coo_new_value[nnz_offset + tmp]=new_coo_value[i];
        coo_new_colidx[nnz_offset + tmp]=new_coo_colidx[i];
//printf("global_id=%d i=%d  row=%d  new_coo_colidx[i]=%d  nnz_offset=%d tmp=%d nnz_offset + tmp=%d  new_coo_colidx[i]=%d  new_coo_value[i]=%f\n",global_id,i,row,new_coo_colidx[i],nnz_offset,tmp,nnz_offset + tmp,new_coo_colidx[i],new_coo_value[i]);
      //  printf("global_id=%d i=%d  new_coo_colidx[i]=%d  new_coo_value[i]=%d\n",global_id,i,new_coo_colidx[i],new_coo_value[i]);
    }
__syncthreads();

}

//cuda_bal_step1<<< num_blocks, 64 >>>( matrixA_d->tilem, matrixA_d->tilen, d_tile_ptr_A,start,end,num_threads,
  //      d_flag_bal_tile_rowidx,d_tile_bal_rowidx_colstart,d_tile_bal_rowidx_colstop,d_group_ptr,d_bal_num_gro);

__global__
void cuda_bal_step1(int tilem, int tilen, MAT_PTR_TYPE *tile_ptr, int start, int end, int num_threads,
       unsigned int *flag_bal_tile_rowidx, int *tile_bal_rowidx_colstart,int *tile_bal_rowidx_colstop, int *bal_num_gro,int tilecnt_ave)
{
   // if(threadIdx.x==1)
    //printf("!!");

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int blki=start+global_id;
    if(global_id<num_threads)
    {   

        int balancenumblk = tile_ptr[blki + 1] - tile_ptr[blki];
        if (balancenumblk <= tilecnt_ave) {
            flag_bal_tile_rowidx[bal_num_gro[blki]] = blki;
            tile_bal_rowidx_colstart[bal_num_gro[blki]] = tile_ptr[blki] ;
            tile_bal_rowidx_colstop[bal_num_gro[blki]] = tile_ptr[blki] + balancenumblk;
//printf("global_id=%d  blki=%d  tile_ptr[blki]=%d  bal_num_gro[blki]=%d\n",global_id,blki,tile_ptr[blki],bal_num_gro[blki]);
            //rowtilekcnt++;
        } 
        else 
        {
            int numblklocal = ceil((double) balancenumblk / (double) tilecnt_ave);
            // printf("numblklocal = %i\n", numblklocal);
            int lenblklocal = ceil((double) balancenumblk / (double) numblklocal);
            // printf("lenblklocal = %i\n", lenblklocal);
//printf("global_id=%d  blki=%d  tile_ptr[blki]=%d  bal_num_gro[blki]=%d\n",global_id,blki,tile_ptr[blki],bal_num_gro[blki]);
            for (int iii = 0; iii < numblklocal; iii++) {
                flag_bal_tile_rowidx[bal_num_gro[blki]+iii] = blki ; // can generate -0
                tile_bal_rowidx_colstart[bal_num_gro[blki]+iii] = tile_ptr[blki] + iii * lenblklocal;
                if (iii == numblklocal - 1)
                    tile_bal_rowidx_colstop[bal_num_gro[blki]+iii] = tile_ptr[blki] + balancenumblk;
                else
                    tile_bal_rowidx_colstop[bal_num_gro[blki]+iii] = tile_ptr[blki] + (iii + 1) * lenblklocal;

                //rowtilekcnt++;
            }
        }
    }
__syncthreads();

}

__inline__ __device__
int d_binary_search_right_boundary_kernel(const int *row_pointer,
                                        const int  key_input,
                                        const int  size)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}

//cuda_bal_step2<<< nthreads/64+1, 64 >>>( stridennz_1, matrixA_d->nnz,nthreads,rowblkblock_1,d_flag_tilerow_start,d_group_ptr);
__global__
void cuda_bal_step2(int stridennz, int nnz, int nthreads, int rowblkblock, int *flag_tilerow_start, int *group_ptr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<nthreads)
    {
        // compute partition boundaries by partition of size stride
	int boundary = tid * stridennz;
	// clamp partition boundaries to [0, nnzR]
	boundary = boundary > nnz ? nnz : boundary;
	// binary search
	flag_tilerow_start[tid] = d_binary_search_right_boundary_kernel(group_ptr, boundary,  rowblkblock+ 1) - 1;
    }
__syncthreads();
}
//d_format_transform(matrixA_d, matrixA,
 //               &new_coo_value_1, &new_coo_colidx_1, &new_coo_rowidx_1, &new_coocount_1,BLOCK_SIZE);

void d_format_transform(Beidou_Tile_Matrix *matrixA_d,  MAT_VAL_TYPE **new_coo_value_temp, int **new_coo_colidx_temp, int **new_coo_rowidx_temp, int **new_coocount_temp)
//(Beidou_Tile_Matrix *matrixA_d, Beidou_Tile_Matrix *matrixA,
                  //MAT_VAL_TYPE **new_coo_value_temp, int **new_coo_colidx_temp, int **new_coo_rowidx_temp, int **new_coocount_1, int BLOCK_SIZE)
{ 
    struct timeval t1, t2;
    int num_threads, num_blocks;
    //num_threads = 32;
    num_blocks = matrixA_d->tilem*matrixA_d->tilen; 
    MAT_PTR_TYPE *d_rowpointerA;
    MAT_PTR_TYPE *d_columnindexA;

    MAT_PTR_TYPE *d_tile_ptr_A;
    int *d_flag_t;
    int n_tile=1024;
    int *flag_t=(int *)malloc(n_tile*matrixA_d->tilen * sizeof(int));
    memset(flag_t,0,n_tile*matrixA_d->tilen * sizeof(int));

    cudaMalloc((void **)&d_flag_t, n_tile*matrixA_d->tilen * sizeof(int) );
    //cudaMemcpy(d_flag_t, flag_t, n_tile*tilenA * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_rowpointerA, sizeof(MAT_PTR_TYPE) *(matrixA_d->m+1) );
    cudaMemcpy(d_rowpointerA, matrixA_d->rowpointer, sizeof(MAT_PTR_TYPE) *(matrixA_d->m+1), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_columnindexA, sizeof(MAT_PTR_TYPE) *(matrixA_d->nnz+1) );
    cudaMemcpy(d_columnindexA, matrixA_d->columnidx, sizeof(MAT_PTR_TYPE) *(matrixA_d->nnz+1), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_tile_ptr_A, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1) );
    cudaMemcpy(d_tile_ptr_A, matrixA_d->tile_ptr, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1), cudaMemcpyHostToDevice);

    double time_cuda_step1 = 0;
    for(int i=0;i<matrixA_d->tilem;i+=n_tile)
    {   
        cudaMemset(d_flag_t,0,n_tile*matrixA_d->tilen * sizeof(int));
        int row_start_idx=i*BLOCK_SIZE;
        int row_end_idx=i*BLOCK_SIZE+n_tile*BLOCK_SIZE>matrixA_d->m-1 ? row_end_idx=matrixA_d->m-1 : row_end_idx=i*BLOCK_SIZE+n_tile*BLOCK_SIZE-1;
        //printf("row_start_idx=%d  row_end_idx=%d\n",row_start_idx,row_end_idx);
        num_threads= matrixA_d->rowpointer[row_end_idx+1]-matrixA_d->rowpointer[row_start_idx];
        num_blocks=num_threads/64+1;
        // num_blocks=num_threads/64;
         //if(num_threads%64!=0)  num_blocks++;
        //num_threads%64==0 ? num_threads=num_threads/64 : num_threads=num_threads/64+1;
        
        gettimeofday(&t1, NULL);
        

        //cuda_step1_kernel<<< num_blocks, num_threads >>>(rowA, colA, d_rowpointerA, d_columnindexA, tilemA, tilenA, d_tile_ptr_A);
        
        cuda_step1_kernel<<< num_blocks, 64 >>>(matrixA_d->m, matrixA_d->n, d_rowpointerA, d_columnindexA, matrixA_d->tilem, matrixA_d->tilen, d_tile_ptr_A,d_flag_t,row_start_idx,row_end_idx,num_threads,n_tile);

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        //int length;
//printf("num_thread %d    num_block=%d\n",num_threads,num_blocks);
        int length=i+n_tile>matrixA_d->tilem ? length=matrixA_d->tilem-i : length=n_tile ;
        //i*BLOCK_SIZE+n_tile*BLOCK_SIZE>rowA-1 ? length=tilemA-i : length=n_tile ;
        //printf("length=%d\n",length);
        cudaMemcpy(matrixA_d->tile_ptr+i, d_tile_ptr_A+i, sizeof(MAT_PTR_TYPE) *(length+1), cudaMemcpyDeviceToHost);
       
        time_cuda_step1 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    }
//printf("\n\n\n");
    printf("transform_step1(cuda) runtime    = %4.5f ms\n", time_cuda_step1);
    //cudaMemcpy(tile_ptr_A_1, d_tile_ptr_A, sizeof(MAT_PTR_TYPE) *(tilemA+1), cudaMemcpyDeviceToHost);

    //cuda-step1-end
    exclusive_scan(matrixA_d->tile_ptr, matrixA_d->tilem+1);
  /*  for(int i=0;i<matrixA_d->tilem;i++)
    {
        if(matrixA->tile_ptr[i]!=matrixA_d->tile_ptr[i])
        { 
             printf("step1-error! i=%d   %d!=%d\n",i,matrixA->tile_ptr[i],matrixA_d->tile_ptr[i]);
             //break;
        }
    }*/
    matrixA_d->numtile = matrixA_d->tile_ptr[matrixA_d->tilem];
//cuda-step2
    cudaMemcpy(d_tile_ptr_A, matrixA_d->tile_ptr, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1), cudaMemcpyHostToDevice);

    int *d_tile_columnidx;
   // int *tile_columnidx_1;                // block columnindex of A
    matrixA_d->tile_columnidx=(int *)malloc((matrixA_d->numtile+1)*sizeof(int));
    memset(matrixA_d->tile_columnidx, 0, (matrixA_d->numtile+1)*sizeof(int));
    cudaMalloc((void **)&d_tile_columnidx, (matrixA_d->numtile+1) * sizeof(int) );
    cudaMemcpy(d_tile_columnidx, matrixA_d->tile_columnidx, (matrixA_d->numtile+1) * sizeof(int), cudaMemcpyHostToDevice);

    //int *tile_nnz_1 = (int *)malloc((numtileA + 1)* sizeof(int)); //real nnz of each sparse tile 
    matrixA_d->tile_nnz=(int *)malloc((matrixA_d->numtile + 1)* sizeof(int));
    int *d_tile_nnz;
    memset(matrixA_d->tile_nnz,0,(matrixA_d->numtile + 1) * sizeof(int));
    cudaMalloc((void **)&d_tile_nnz, (matrixA_d->numtile+1) * sizeof(int) );
    cudaMemcpy(d_tile_nnz, matrixA_d->tile_nnz, (matrixA_d->numtile+1) * sizeof(int), cudaMemcpyHostToDevice);

    int *d_tile_csr_ptr;
    matrixA_d->csr_ptr_1 = (int *)malloc(((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(int));
   // matrixA_d->csr_ptr = (unsigned char *)malloc(((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(unsigned char));
    memset (matrixA_d->csr_ptr_1, 0, ((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(int));
    cudaMalloc((void **)&d_tile_csr_ptr, ((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(int) );
    cudaMemcpy(d_tile_csr_ptr, matrixA_d->csr_ptr_1, ((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(int), cudaMemcpyHostToDevice);

      int *d_j_col;
    int *j_col=(int *)malloc((n_tile+1)*sizeof(int));
    memset(j_col, 0, (n_tile+1)*sizeof(int));
    cudaMalloc((void **)&d_j_col, (n_tile+1) * sizeof(int) );
  //  cudaMemcpy(d_j_col, j_col, (n_tile+1) * sizeof(int), cudaMemcpyHostToDevice);

   int *d_j_num_t_1;
    int *j_num_t_1=(int *)malloc(n_tile*matrixA_d->tilen * sizeof(int));
    memset(j_num_t_1,0,n_tile*matrixA_d->tilen * sizeof(int));
   cudaMalloc((void **)&d_j_num_t_1, n_tile*matrixA_d->tilen * sizeof(int) );
 //   cudaMemcpy(d_j_num_t_1, j_num_t_1, n_tile*tilenA * sizeof(int), cudaMemcpyHostToDevice);  
printf("matrixA_d->tilem=%d\n",matrixA_d->tilem);
n_tile=1024;


    double time_cuda_step2 = 0;
    for(int i=0;i<matrixA_d->tilem;i+=n_tile)
    {   //printf("hhhh\n");
        cudaMemset(d_j_col,0,(n_tile+1) * sizeof(int));
        cudaMemset(d_flag_t,0,n_tile*matrixA_d->tilen * sizeof(int));
        cudaMemset(d_j_num_t_1,-1,n_tile*matrixA_d->tilen * sizeof(int));
        int row_start_idx=i*BLOCK_SIZE;
        int row_end_idx=i*BLOCK_SIZE+n_tile*BLOCK_SIZE>matrixA_d->m-1 ? row_end_idx=matrixA_d->m-1 : row_end_idx=i*BLOCK_SIZE+n_tile*BLOCK_SIZE-1;
        //printf("row_start_idx=%d  row_end_idx=%d\n",row_start_idx,row_end_idx);
        num_threads= matrixA_d->rowpointer[row_end_idx+1]-matrixA_d->rowpointer[row_start_idx];
        num_blocks=num_threads/64+1;
        // num_blocks=num_threads/64;
         //if(num_threads%64!=0)  num_blocks++;
        //num_threads%64==0 ? num_threads=num_threads/64 : num_threads=num_threads/64+1;
        
        gettimeofday(&t1, NULL);
        //printf("num_thread %d    num_block=%d\n",num_threads,num_blocks);

        //cuda_step1_kernel<<< num_blocks, num_threads >>>(rowA, colA, d_rowpointerA, d_columnindexA, tilemA, tilenA, d_tile_ptr_A);
        
        cuda_step2_kernel<<< num_blocks, 64 >>>(matrixA_d->m, matrixA_d->n, d_rowpointerA, d_columnindexA, matrixA_d->tilem, matrixA_d->tilen, num_threads, row_start_idx, row_end_idx, d_tile_ptr_A,d_tile_columnidx,d_tile_nnz,d_tile_csr_ptr, matrixA_d->numtile,d_j_col,n_tile, d_flag_t,d_j_num_t_1);
        cudaDeviceSynchronize();
cuda_step2_2_kernel<<< num_blocks, 64 >>>(matrixA_d->m, matrixA_d->n, d_rowpointerA, d_columnindexA, matrixA_d->tilem, matrixA_d->tilen, num_threads, row_start_idx, row_end_idx, d_tile_ptr_A,d_tile_columnidx,d_tile_nnz,d_tile_csr_ptr, matrixA_d->numtile,d_j_col,n_tile, d_flag_t,d_j_num_t_1);
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        //int length;
        //int length=i+n_tile>tilemA ? length=tilemA-i : length=n_tile ;
        int length=matrixA_d->tile_ptr[row_end_idx/BLOCK_SIZE+1]-matrixA_d->tile_ptr[i];
       // printf("i=%d  row_end_idx/BLOCK_SIZE+1=%d  tile_ptr_A[row_end_idx/BLOCK_SIZE+1]=%d  tile_ptr_A[i]=%d  -:%d\n",i,row_end_idx/BLOCK_SIZE+1,tile_ptr_A[row_end_idx/BLOCK_SIZE+1],tile_ptr_A[i],tile_ptr_A[row_end_idx/BLOCK_SIZE+1]-tile_ptr_A[i]);
        //printf("length=%d\n",length);
        //printf("tile_ptr_A[i]=%d\n",tile_ptr_A[i]);
        //cudaMemcpy(tile_nnz_1+tile_ptr_A[i], d_tile_nnz+tile_ptr_A[i], sizeof(MAT_PTR_TYPE) *(length+1), cudaMemcpyDeviceToHost);
        //cudaMemcpy(tile_columnidx_1+tile_ptr_A[i], d_tile_columnidx+tile_ptr_A[i], sizeof(MAT_PTR_TYPE) *(length+1), cudaMemcpyDeviceToHost);
        //cudaMemcpy(tile_csr_ptr_1+(tile_ptr_A[i]*BLOCK_SIZE), d_tile_csr_ptr+(tile_ptr_A[i]*BLOCK_SIZE), sizeof(MAT_PTR_TYPE) *((length*BLOCK_SIZE)+1), cudaMemcpyDeviceToHost);
        time_cuda_step2 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    }
    cudaMemcpy(matrixA_d->tile_columnidx, d_tile_columnidx, matrixA_d->numtile*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->tile_nnz, d_tile_nnz, matrixA_d->numtile*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->csr_ptr_1, d_tile_csr_ptr, BLOCK_SIZE*(matrixA_d->numtile + 1)*sizeof(int), cudaMemcpyDeviceToHost);
    printf("transform_step2(cuda) runtime    = %4.5f ms\n", time_cuda_step2);
    for(int blki =0;blki < matrixA_d->tilem ;blki ++)
    {
         //quick_sort_key(tile_columnidx_1 + tile_ptr_A[blki],tile_ptr_A[blki+1] - tile_ptr_A[blki]);
         //quick_sort_key_val_pair(tile_columnidx_1 + tile_ptr_A[blki], tile_nnz_1 + tile_ptr_A[blki],tile_ptr_A[blki+1] - tile_ptr_A[blki]);
        quick_sort_key_tile(matrixA_d->tile_columnidx + matrixA_d->tile_ptr[blki],matrixA_d->tile_ptr[blki+1] - matrixA_d->tile_ptr[blki], matrixA_d->tile_nnz + matrixA_d->tile_ptr[blki], matrixA_d->csr_ptr_1+matrixA_d->tile_ptr[blki]*BLOCK_SIZE);
    }
    exclusive_scan(matrixA_d->tile_nnz, matrixA_d->numtile +1);
    
  /*  for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA_d->tile_columnidx[i]!=matrixA->tile_columnidx[i]) 
        {
             printf("step2-error-colidx! i=%d   %d!=%d\n",i,matrixA->tile_columnidx[i],matrixA_d->tile_columnidx[i]);
             break;
        }
        if(matrixA_d->tile_nnz[i]!=matrixA->tile_nnz[i]) 
        {
             printf("step2-error-nnz! i=%d   %d!=%d\n",i,matrixA->tile_nnz[i],matrixA_d->tile_nnz[i]);
             break;
        }
        for(int j=0;j<BLOCK_SIZE;j++)
        {
            if(matrixA->csr_ptr[i*BLOCK_SIZE+j]!=matrixA_d->csr_ptr_1[i*BLOCK_SIZE+j])
            {
               // printf("step2-error-ptr! i=%d   j=%d   %d!=%d\n",i,j,csr_ptr[i*BLOCK_SIZE+j],matrixA_d->csr_ptr_1[i*BLOCK_SIZE+j]);
                break;
            }
        }
        //printf("\n");
    }*/
    
//cuda-step2-end

//cuda-step3
   //format 0-7 represent 7 formats: CSR, COO, ELL, HYB, Dns, DnsRow, DnsCol
    //char *Format_1 =(char *)malloc(numtileA* sizeof(char));
    matrixA_d->Format =(char *)malloc(matrixA_d->numtile* sizeof(char));
    memset(matrixA_d->Format,0,matrixA_d->numtile * sizeof(char));
    char *d_Format;
    cudaMalloc((void **)&d_Format, matrixA_d->numtile* sizeof(char) );
    cudaMemcpy(d_Format, matrixA_d->Format, matrixA_d->numtile* sizeof(char), cudaMemcpyHostToDevice);

    //int *blknnz_1 = (int *)malloc((numtileA + 1)* sizeof(int)); //space cost that need allocate 
    matrixA_d->blknnz = (int *)malloc((matrixA_d->numtile + 1)* sizeof(int));                                               
    memset(matrixA_d->blknnz,0,(matrixA_d->numtile + 1) * sizeof(int));  
    int *d_blknnz;
    cudaMalloc((void **)&d_blknnz, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_blknnz, matrixA_d->blknnz, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

 


    //dense 
    int dense_size_1=0;
    matrixA_d->dns_offset = (int *)malloc((matrixA_d->numtile+1) * sizeof(int));
    memset(matrixA_d->dns_offset, 0, (matrixA_d->numtile+1) * sizeof(int));
    int *d_dns_offset;
    cudaMalloc((void **)&d_dns_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_dns_offset, matrixA_d->dns_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);


    //denserow
     matrixA_d->denserowptr = (int *)malloc((matrixA_d->numtile + 1) * sizeof(int));
    memset(matrixA_d->denserowptr,0,(matrixA_d->numtile+ 1) * sizeof(int));
    int *d_denserowptr;
   cudaMalloc((void **)&d_denserowptr, (matrixA_d->numtile + 1)* sizeof(int) );
   // cudaMemcpy(d_denserowptr, denserowptr_1, (numtileA + 1)* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_denserowptr,0,(matrixA_d->numtile+ 1) * sizeof(int));
 
    int denserow_size_1 =0 ;
    matrixA_d->dnsrow_offset = (int *)malloc((matrixA_d->numtile+1) * sizeof(int));
    memset(matrixA_d->dnsrow_offset, 0, (matrixA_d->numtile+1) * sizeof(int));
    int *d_dnsrow_offset;
    cudaMalloc((void **)&d_dnsrow_offset, (matrixA_d->numtile + 1)* sizeof(int) );
   // cudaMemcpy(d_dnsrow_offset, dnsrow_offset_1, (numtileA + 1)* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_dnsrow_offset,0, (matrixA_d->numtile+1) * sizeof(int));
    
 
     //densecolumn
    matrixA_d->densecolptr = (int *)malloc((matrixA_d->numtile + 1) * sizeof(int));
    memset(matrixA_d->densecolptr,0,(matrixA_d->numtile+ 1) * sizeof(int));
    int *d_densecolptr;
    cudaMalloc((void **)&d_densecolptr, (matrixA_d->numtile + 1)* sizeof(int) );
  //  cudaMemcpy(d_densecolptr, densecolptr_1, (numtileA + 1)* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_densecolptr,0,(matrixA_d->numtile+ 1) * sizeof(int));
   
    int densecol_size_1 =0 ;
    matrixA_d->dnscol_offset = (int *)malloc((matrixA_d->numtile+1) * sizeof(int));
    memset(matrixA_d->dnscol_offset, 0, (matrixA_d->numtile+1) * sizeof(int));
    int *d_dnscol_offset;
    cudaMalloc((void **)&d_dnscol_offset, (matrixA_d->numtile + 1)* sizeof(int) );
  //  cudaMemcpy(d_dnscol_offset, dnscol_offset_1, (numtileA + 1)* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_dnscol_offset,0,(matrixA_d->numtile+ 1) * sizeof(int));
     //CSR
    
    int csrsize_1=0;
  //  int csrptrlen=0;
    matrixA_d->csr_offset = (int *)malloc((matrixA_d->numtile+1) * sizeof(int));
    memset(matrixA_d->csr_offset, 0, (matrixA_d->numtile+1) * sizeof(int)); 
    int *d_csr_offset;
    cudaMalloc((void **)&d_csr_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_csr_offset, matrixA_d->csr_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    matrixA_d->csrptr_offset = (int *)malloc((matrixA_d->numtile+1) * sizeof(int));
    memset(matrixA_d->csrptr_offset, 0, (matrixA_d->numtile+1) * sizeof(int));
    int *d_csrptr_offset;
    cudaMalloc((void **)&d_csrptr_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_csrptr_offset, matrixA_d->csrptr_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    //ELL
    matrixA_d->ell_offset = (int *)malloc((matrixA_d->numtile+1) * sizeof(int));
    memset(matrixA_d->ell_offset, 0, (matrixA_d->numtile+1) * sizeof(int));
    int ellsize_1 =0;
    int *d_ell_offset;
    cudaMalloc((void **)&d_ell_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_ell_offset, matrixA_d->ell_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);


    //COO
    int coosize_1 =0;
    matrixA_d->coo_offset = (int *)malloc((matrixA_d->numtile+1) * sizeof(int));
    memset(matrixA_d->coo_offset, 0, (matrixA_d->numtile+1) * sizeof(int));
    int *d_coo_offset;
    cudaMalloc((void **)&d_coo_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_coo_offset, matrixA_d->coo_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    //HYB
    int hybellsize_1 =0;
    int hybcoosize_1 =0;
    int hybsize_1 =0;
    matrixA_d->blkwidth = (char *)malloc(matrixA_d->numtile*sizeof(char));
    memset(matrixA_d->blkwidth,0,matrixA_d->numtile * sizeof(char)) ;
    char *d_blkwidth;
    cudaMalloc((void **)&d_blkwidth, (matrixA_d->numtile + 1)* sizeof(char) );
    cudaMemcpy(d_blkwidth, matrixA_d->blkwidth, (matrixA_d->numtile + 1)* sizeof(char), cudaMemcpyHostToDevice);


    matrixA_d->hyb_coocount= (int *)malloc((matrixA_d->numtile + 1) * sizeof(int));
    memset(matrixA_d->hyb_coocount,0,(matrixA_d->numtile + 1) * sizeof(int)) ;
    int *d_hyb_coocount;
    cudaMalloc((void **)&d_hyb_coocount, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_hyb_coocount, matrixA_d->hyb_coocount, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);


    matrixA_d->hyb_offset = (int *)malloc((matrixA_d->numtile+1) * sizeof(int));
    memset(matrixA_d->hyb_offset, 0, (matrixA_d->numtile+1) * sizeof(int));
    int *d_hyb_offset;
    cudaMalloc((void **)&d_hyb_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_hyb_offset, matrixA_d->hyb_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    *new_coocount_temp = (int *)malloc((matrixA_d->numtile + 1) * sizeof(int));
    memset(*new_coocount_temp,0,(matrixA_d->numtile + 1) * sizeof(int)) ;
    int *new_coocount_1 = *new_coocount_temp;

    int *d_new_coocount;
    cudaMalloc((void **)&d_new_coocount, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_new_coocount, new_coocount_1, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);


int num_tile_row=1024;
    unsigned char *col_flag =(unsigned char *)malloc(matrixA_d->tilen*num_tile_row*BLOCK_SIZE * sizeof(unsigned char));
    memset(col_flag, 0, matrixA_d->tilen*num_tile_row*BLOCK_SIZE * sizeof(unsigned char));
    unsigned char *d_col_flag;
    cudaMalloc((void **)&d_col_flag, matrixA_d->tilen*num_tile_row*BLOCK_SIZE * sizeof(unsigned char) );
    cudaMemcpy(d_col_flag, col_flag, matrixA_d->tilen*num_tile_row*BLOCK_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);

cudaMemcpy(d_tile_csr_ptr, matrixA_d->csr_ptr_1, BLOCK_SIZE*(matrixA_d->numtile + 1)*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_rowpointerA, matrixA_d->rowpointer, sizeof(MAT_PTR_TYPE) *(matrixA_d->m+1), cudaMemcpyHostToDevice);
cudaMemcpy(d_tile_columnidx, matrixA_d->tile_columnidx, matrixA_d->numtile*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_tile_nnz, matrixA_d->tile_nnz, (matrixA_d->numtile+1)*sizeof(int), cudaMemcpyHostToDevice);
//num_blocks=32;

//printf("tilemA=%d\n",tilemA);
    gettimeofday(&t1, NULL);
   // int x=0;
    for (int blki=0;blki<matrixA_d->tilem;blki+=num_tile_row)//
    {
//int blki=0;
        cudaMemset(d_col_flag,0,matrixA_d->tilen*num_tile_row*BLOCK_SIZE * sizeof(unsigned char));
        int start=blki;
        int end= blki+num_tile_row<matrixA_d->tilem ? end= blki+num_tile_row : end=matrixA_d->tilem;
        num_blocks=matrixA_d->tile_ptr[end]-matrixA_d->tile_ptr[start];
        //printf("end=%d  start=%d  num_blocks=%d  blki=%d\n",end,start,num_blocks,blki);
      //  int tilenum_per_row=tile_ptr_A[blki+1]-tile_ptr_A[blki];
        
  //      int rowlen= blki==tilemA-1 ? rowA-(tilemA-1)*BLOCK_SIZE : BLOCK_SIZE ;
        cuda_step3_kernel<<<num_blocks, 32 >>>(matrixA_d->m, matrixA_d->n,d_rowpointerA, d_columnindexA,
                 matrixA_d->tilem, matrixA_d->tilen, matrixA_d->numtile, d_tile_ptr_A, d_tile_columnidx, d_tile_nnz, d_Format, 
                 d_tile_csr_ptr, d_blknnz, d_blkwidth, d_hyb_coocount,
                 d_denserowptr, d_densecolptr,
                 d_csr_offset, d_csrptr_offset, d_coo_offset, d_ell_offset, d_hyb_offset, d_dns_offset, d_dnsrow_offset, d_dnscol_offset,blki,num_tile_row,d_col_flag,d_new_coocount); 
        
    }
    gettimeofday(&t2, NULL);
    cudaDeviceSynchronize();
    double cuda_time_step3  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("cuda_transform_step3 runtime    = %4.5f ms\n", cuda_time_step3);
  
	cudaMemcpy(matrixA_d->Format, d_Format, matrixA_d->numtile* sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrixA_d->blknnz, d_blknnz, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);
printf("numtileA=%d\n",matrixA_d->numtile);

	cudaMemcpy(matrixA_d->csr_offset, d_csr_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrixA_d->csrptr_offset, d_csrptr_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(matrixA_d->dns_offset, d_dns_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(matrixA_d->denserowptr, d_denserowptr, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrixA_d->dnsrow_offset, d_dnsrow_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(matrixA_d->dnscol_offset, d_dnscol_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrixA_d->densecolptr, d_densecolptr, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(matrixA_d->ell_offset, d_ell_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(matrixA_d->coo_offset, d_coo_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(matrixA_d->hyb_offset, d_hyb_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrixA_d->hyb_coocount, d_hyb_coocount, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(matrixA_d->blkwidth, d_blkwidth, (matrixA_d->numtile+1)* sizeof(char), cudaMemcpyDeviceToHost);

        cudaMemcpy(new_coocount_1, d_new_coocount, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyDeviceToHost);

    exclusive_scan(matrixA_d->csr_offset, matrixA_d->numtile +1);
    exclusive_scan(matrixA_d->csrptr_offset, matrixA_d->numtile +1);
    exclusive_scan(matrixA_d->coo_offset, matrixA_d->numtile +1);
    exclusive_scan(matrixA_d->ell_offset, matrixA_d->numtile +1);
    exclusive_scan(matrixA_d->hyb_offset, matrixA_d->numtile +1);
    exclusive_scan(matrixA_d->dns_offset, matrixA_d->numtile +1);
    exclusive_scan(matrixA_d->dnsrow_offset, matrixA_d->numtile +1);
    exclusive_scan(matrixA_d->dnscol_offset, matrixA_d->numtile +1);
    exclusive_scan(matrixA_d->denserowptr,matrixA_d->numtile+1);
    exclusive_scan(matrixA_d->densecolptr,matrixA_d->numtile+1);
    exclusive_scan(matrixA_d->hyb_coocount, matrixA_d->numtile +1);
    hybcoosize_1 = matrixA_d->hyb_coocount[matrixA_d->numtile];
    exclusive_scan(new_coocount_1, matrixA_d->numtile +1);

    for (int blki=0;blki<matrixA_d->tilem;blki++)
    {
        int rowlength= blki==matrixA_d->tilem-1 ? matrixA_d->m-(matrixA_d->tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int rowbnum=matrixA_d->tile_ptr[blki+1]-matrixA_d->tile_ptr[blki];
        for (int bi=0;bi<rowbnum;bi++)
        {
            char format= matrixA_d->Format[matrixA_d->tile_ptr[blki]+bi];
            switch (format)
            {
                case 0:    //csr
                    csrsize_1 +=  matrixA_d->blknnz[matrixA_d->tile_ptr[blki]+bi];
            //        csrptrlen += rowlength ;
                    break;
                
                case 1:  //coo
                    coosize_1 += matrixA_d->blknnz[matrixA_d->tile_ptr[blki]+bi];
                    break;
                case 2:  //ell
                    ellsize_1 += matrixA_d->blknnz[matrixA_d->tile_ptr[blki]+bi] ;
                    break;
                case 3: //hyb
                    hybsize_1 += matrixA_d->blknnz[matrixA_d->tile_ptr[blki]+bi];
                    hybellsize_1 += matrixA_d->blkwidth[matrixA_d->tile_ptr[blki]+bi] * rowlength;
                    break;
                case 4:
                    dense_size_1 += matrixA_d->blknnz[matrixA_d->tile_ptr[blki]+bi];
                    break;
                case 5:
                    denserow_size_1 += matrixA_d->blknnz[matrixA_d->tile_ptr[blki]+bi];
                    break;
                case 6:
                    densecol_size_1 += matrixA_d->blknnz[matrixA_d->tile_ptr[blki]+bi];
                    break;
            
                default:
                    break;
            }

        }
    }   
    matrixA_d->blknnznnz = (unsigned char *)malloc((matrixA_d->numtile + 1)* sizeof(unsigned char));
   // memcpy(matrixA_d->blknnznnz,matrixA_d->blknnz,matrixA_d->numtile + 1);
    for (int i = 0; i < matrixA_d->numtile+1; i++)
        matrixA_d->blknnznnz[i] = matrixA_d->blknnz[i];
    //exclusive_scan(blknnz,(numtileA+1));
    exclusive_scan(matrixA_d->blknnz,(matrixA_d->numtile+1));

    int *formatnum_1 = (int *)malloc(7 * sizeof(int));
    memset(formatnum_1,0,7 * sizeof(int));

    for (int j=0;j<7;j++)
    {
        for (int i=0;i<matrixA_d->numtile;i++)
        {
            if (matrixA_d->Format[i]==j)
            {
                formatnum_1[j]++;
             //   printf("%d   ",Format[i]);
             //   break ;
            }
        }
    }

   for (int j=0;j<7;j++)
    {
        printf("format =%i,count =%i\n",j,formatnum_1[j]);
    }
    
    int csrtilecount_1 = formatnum_1[0];
    matrixA_d->csrtilecount=csrtilecount_1;
    /*     for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->Format[i]!=matrixA_d->Format[i])
        {
            printf("step3-error-Format! i=%d   %d!=%d\n",i,matrixA->Format[i],matrixA_d->Format[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->blknnz[i]!=matrixA_d->blknnz[i])
        {
            printf("step3-error-blknnz! i=%d   %d!=%d\n",i,matrixA->blknnz[i],matrixA_d->blknnz[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->csr_offset[i]!=matrixA_d->csr_offset[i])
        {
            printf("step3-error-csr_offset! i=%d   %d!=%d\n",i,matrixA->csr_offset[i],matrixA_d->csr_offset[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->csrptr_offset[i]!=matrixA_d->csrptr_offset[i])
        {
            printf("step3-error-csrptr_offset! i=%d   %d!=%d\n",i,matrixA->csrptr_offset[i],matrixA_d->csrptr_offset[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->dns_offset[i]!=matrixA_d->dns_offset[i])
        {
            printf("step3-error-dns_offset! i=%d   %d!=%d\n",i,matrixA->dns_offset[i],matrixA_d->dns_offset[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->denserowptr[i]!=matrixA_d->denserowptr[i])
        {
            printf("step3-error-denserowptr! i=%d   %d!=%d\n",i,matrixA->denserowptr[i],matrixA_d->denserowptr[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->dnsrow_offset[i]!=matrixA_d->dnsrow_offset[i])
        {
            printf("step3-error-dnsrow_offset! i=%d   %d!=%d\n",i,matrixA->dnsrow_offset[i],matrixA_d->dnsrow_offset[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->dnscol_offset[i]!=matrixA_d->dnscol_offset[i])
        {
            printf("step3-error-dnscol_offset! i=%d   %d!=%d\n",i,matrixA->dnscol_offset[i],matrixA_d->dnscol_offset[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->densecolptr[i]!=matrixA_d->densecolptr[i])
        {
            printf("step3-error-densecolptr! i=%d   %d!=%d\n",i,matrixA->densecolptr[i],matrixA_d->densecolptr[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->ell_offset[i]!=matrixA_d->ell_offset[i])
        {
            printf("step3-error-ell_offset! i=%d   %d!=%d\n",i,matrixA->ell_offset[i],matrixA_d->ell_offset[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->coo_offset[i]!=matrixA_d->coo_offset[i])
        {
            printf("step3-error-coo_offset! i=%d   %d!=%d\n",i,matrixA->coo_offset[i],matrixA_d->coo_offset[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->hyb_offset[i]!=matrixA_d->hyb_offset[i])
        {
            printf("step3-error-hyb_offset! i=%d   %d!=%d\n",i,matrixA->hyb_offset[i],matrixA_d->hyb_offset[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->hyb_coocount[i]!=matrixA_d->hyb_coocount[i])
        {
            printf("step3-error-hyb_coocount! i=%d   %d!=%d\n",i,matrixA->hyb_coocount[i],matrixA_d->hyb_coocount[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)
    {
        if(matrixA->blkwidth[i]!=matrixA_d->blkwidth[i])
        {
            printf("step3-error-blkwidth! i=%d   %d!=%d\n",i,matrixA->blkwidth[i],matrixA_d->blkwidth[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->numtile;i++)//
    {
    //   if(new_coocount[i]!=new_coocount_1[i])
    //    {
    //        printf("step3-error-new_coocount! i=%d   %d!=%d\n",i,new_coocount[i],new_coocount_1[i]);
     //       break;
    //    }
    }*/
matrixA_d->coocount = hybcoosize_1 + coosize_1;
int nnz_temp =0;
    int tile_count_temp =0;
    for (int blki =0;blki < matrixA_d->tilem; blki ++)
    {
        int start= blki*BLOCK_SIZE;
        int end = blki==matrixA_d->tilem-1 ?  matrixA_d->m : (blki+1)*BLOCK_SIZE ;
        nnz_temp = nnz_temp < matrixA_d->rowpointer[end] - matrixA_d->rowpointer[start] ? matrixA_d->rowpointer[end] - matrixA_d->rowpointer[start] : nnz_temp;
        tile_count_temp = tile_count_temp < matrixA_d->tile_ptr[blki +1] - matrixA_d->tile_ptr[blki] ? matrixA_d->tile_ptr[blki +1] - matrixA_d->tile_ptr[blki] : tile_count_temp;
    }

//cuda-step3-end

//cuda-step4

    matrixA_d->csrsize=csrsize_1;
    matrixA_d->coosize=coosize_1;
    matrixA_d->ellsize=ellsize_1;
    matrixA_d->hybsize=hybsize_1;
    matrixA_d->hybellsize=hybellsize_1;
    matrixA_d->dense_size=dense_size_1;
    matrixA_d->denserow_size=denserow_size_1;
    matrixA_d->densecol_size=densecol_size_1;
matrixA_d->hybcoosize=hybcoosize_1;
    
     //CSR
    //MAT_VAL_TYPE *Tile_csr_Val_1=(MAT_VAL_TYPE*)malloc((csrsize_1)*sizeof(MAT_VAL_TYPE));
    matrixA_d->Tile_csr_Val=(MAT_VAL_TYPE*)malloc((csrsize_1)*sizeof(MAT_VAL_TYPE));
    memset(matrixA_d->Tile_csr_Val, 0, (csrsize_1)*sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_Tile_csr_Val;
    cudaMalloc((void **)&d_Tile_csr_Val, (csrsize_1)*sizeof(MAT_VAL_TYPE) );
    cudaMemcpy(d_Tile_csr_Val, matrixA_d->Tile_csr_Val, (csrsize_1)*sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    //unsigned char  *Tile_csr_Col_1=(unsigned char*)malloc((csrsize_1)*sizeof(unsigned char));
    matrixA_d->Tile_csr_Col=(unsigned char*)malloc((csrsize_1)*sizeof(unsigned char));
    memset(matrixA_d->Tile_csr_Col, 0, (csrsize_1)*sizeof(unsigned char));
    unsigned char  *d_Tile_csr_Col;
    cudaMalloc((void **)&d_Tile_csr_Col,(csrsize_1)*sizeof(unsigned char) );
    cudaMemcpy(d_Tile_csr_Col, matrixA_d->Tile_csr_Col, (csrsize_1)*sizeof(unsigned char), cudaMemcpyHostToDevice);
//printf("csrtilecount=%d  csrtilecount_1=%d\n",csrtilecount,csrtilecount_1);

    //unsigned char *Tile_csr_Ptr_1=(unsigned char*)malloc((csrtilecount_1 * BLOCK_SIZE)*sizeof(unsigned char));
    matrixA_d->Tile_csr_Ptr=(unsigned char*)malloc((csrtilecount_1 * BLOCK_SIZE)*sizeof(unsigned char));
    memset(matrixA_d->Tile_csr_Ptr, 0, (csrtilecount_1 * BLOCK_SIZE )*sizeof(unsigned char));
    unsigned char *d_Tile_csr_Ptr;
    cudaMalloc((void **)&d_Tile_csr_Ptr,(csrtilecount_1 * BLOCK_SIZE)*sizeof(unsigned char) );
    cudaMemcpy(d_Tile_csr_Ptr, matrixA_d->Tile_csr_Ptr,(csrtilecount_1 * BLOCK_SIZE)*sizeof(unsigned char), cudaMemcpyHostToDevice);

    //COO
    //MAT_VAL_TYPE *Tile_coo_Val_1=(MAT_VAL_TYPE*)malloc((coosize_1)*sizeof(MAT_VAL_TYPE));
    matrixA_d->Tile_coo_Val=(MAT_VAL_TYPE*)malloc((coosize_1)*sizeof(MAT_VAL_TYPE));
    memset(matrixA_d->Tile_coo_Val, 0, (coosize_1)*sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_Tile_coo_Val;
    cudaMalloc((void **)&d_Tile_coo_Val, (coosize_1)*sizeof(MAT_VAL_TYPE) );
    cudaMemcpy(d_Tile_coo_Val, matrixA_d->Tile_coo_Val, (coosize_1)*sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    //unsigned char *Tile_coo_colIdx_1=(unsigned char*)malloc((coosize_1)*sizeof(unsigned char));
    matrixA_d->Tile_coo_colIdx=(unsigned char*)malloc((coosize_1)*sizeof(unsigned char));
    memset(matrixA_d->Tile_coo_colIdx, 0, (coosize_1)*sizeof(unsigned char));
    unsigned char *d_Tile_coo_colIdx;
    cudaMalloc((void **)&d_Tile_coo_colIdx, (coosize_1)*sizeof(unsigned char));
    cudaMemcpy(d_Tile_coo_colIdx, matrixA_d->Tile_coo_colIdx, (coosize_1)*sizeof(unsigned char), cudaMemcpyHostToDevice);    

    //unsigned char *Tile_coo_rowIdx_1=(unsigned char*)malloc((coosize_1)*sizeof(unsigned char));
    matrixA_d->Tile_coo_rowIdx=(unsigned char*)malloc((coosize_1)*sizeof(unsigned char));
    memset(matrixA_d->Tile_coo_rowIdx, 0, (coosize_1)*sizeof(unsigned char));
    unsigned char *d_Tile_coo_rowIdx;
    cudaMalloc((void **)&d_Tile_coo_rowIdx, (coosize_1)*sizeof(unsigned char));
    cudaMemcpy(d_Tile_coo_rowIdx, matrixA_d->Tile_coo_rowIdx, (coosize_1)*sizeof(unsigned char), cudaMemcpyHostToDevice);

     //ELL
   // MAT_VAL_TYPE *Tile_ell_Val_1=(MAT_VAL_TYPE*)malloc((ellsize_1)*sizeof(MAT_VAL_TYPE));
    matrixA_d->Tile_ell_Val=(MAT_VAL_TYPE*)malloc((ellsize_1)*sizeof(MAT_VAL_TYPE));
    memset(matrixA_d->Tile_ell_Val,0,(ellsize_1)*sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_Tile_ell_Val;
    cudaMalloc((void **)&d_Tile_ell_Val, (ellsize_1)*sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_Tile_ell_Val, matrixA_d->Tile_ell_Val, (ellsize_1)*sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

  //  unsigned char *Tile_ell_colIdx_1=(unsigned char*)malloc((ellsize_1)*sizeof(unsigned char));
    matrixA_d->Tile_ell_colIdx=(unsigned char*)malloc((ellsize_1)*sizeof(unsigned char));
    memset(matrixA_d->Tile_ell_colIdx, 0, (ellsize_1)*sizeof(unsigned char));
    unsigned char *d_Tile_ell_colIdx;
    cudaMalloc((void **)&d_Tile_ell_colIdx, (ellsize_1)*sizeof(unsigned char));
    cudaMemcpy(d_Tile_ell_colIdx, matrixA_d->Tile_ell_colIdx, (ellsize_1)*sizeof(unsigned char), cudaMemcpyHostToDevice);


     //HYB
    //MAT_VAL_TYPE *Tile_hyb_Val_1=(MAT_VAL_TYPE*)malloc((hybellsize_1+hybcoosize_1)*sizeof(MAT_VAL_TYPE));
    matrixA_d->Tile_hyb_Val=(MAT_VAL_TYPE*)malloc((hybellsize_1+hybcoosize_1)*sizeof(MAT_VAL_TYPE));
    memset(matrixA_d->Tile_hyb_Val,0,(hybellsize_1+hybcoosize_1)*sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_Tile_hyb_Val;
    cudaMalloc((void **)&d_Tile_hyb_Val, (hybellsize_1+hybcoosize_1)*sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_Tile_hyb_Val, matrixA_d->Tile_hyb_Val, (hybellsize_1+hybcoosize_1)*sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

   // unsigned char *Tile_hyb_ellcolIdx_1=(unsigned char*)malloc((hybellsize_1+hybcoosize_1)*sizeof(unsigned char));
    matrixA_d->Tile_hyb_ellcolIdx=(unsigned char*)malloc((hybellsize_1+hybcoosize_1)*sizeof(unsigned char));
    memset(matrixA_d->Tile_hyb_ellcolIdx, 0, (hybellsize_1+hybcoosize_1)*sizeof(unsigned char));
    unsigned char *d_Tile_hyb_ellcolIdx;
    cudaMalloc((void **)&d_Tile_hyb_ellcolIdx, (hybellsize_1+hybcoosize_1)*sizeof(unsigned char));
    cudaMemcpy(d_Tile_hyb_ellcolIdx, matrixA_d->Tile_hyb_ellcolIdx, (hybellsize_1+hybcoosize_1)*sizeof(unsigned char), cudaMemcpyHostToDevice);

    //unsigned char *Tile_hyb_coorowIdx_1=(unsigned char*)malloc((hybcoosize_1)*sizeof(unsigned char)) ;
    matrixA_d->Tile_hyb_coorowIdx=(unsigned char*)malloc((hybcoosize_1)*sizeof(unsigned char)) ;
    memset(matrixA_d->Tile_hyb_coorowIdx, 0, (hybcoosize_1)*sizeof(unsigned char));
    unsigned char *d_Tile_hyb_coorowIdx;
    cudaMalloc((void **)&d_Tile_hyb_coorowIdx, (hybcoosize_1)*sizeof(unsigned char));
    cudaMemcpy(d_Tile_hyb_coorowIdx, matrixA_d->Tile_hyb_coorowIdx, (hybcoosize_1)*sizeof(unsigned char), cudaMemcpyHostToDevice);

     //dense
    //MAT_VAL_TYPE *Tile_dns_Val_1=(MAT_VAL_TYPE*)malloc((dense_size_1)*sizeof(MAT_VAL_TYPE));
    matrixA_d->Tile_dns_Val=(MAT_VAL_TYPE*)malloc((dense_size_1)*sizeof(MAT_VAL_TYPE));
    memset(matrixA_d->Tile_dns_Val,0,(dense_size_1)*sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_Tile_dns_Val;
    cudaMalloc((void **)&d_Tile_dns_Val, (dense_size_1)*sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_Tile_dns_Val, matrixA_d->Tile_dns_Val, (dense_size_1)*sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    //dense row
    //MAT_VAL_TYPE *Tile_denserow_Val_1=(MAT_VAL_TYPE*)malloc((denserow_size_1) * sizeof(MAT_VAL_TYPE));
    matrixA_d->Tile_dnsrow_Val=(MAT_VAL_TYPE*)malloc((denserow_size_1) * sizeof(MAT_VAL_TYPE));
    memset(matrixA_d->Tile_dnsrow_Val,0,(denserow_size_1) * sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_Tile_dnsrow_Val;
    cudaMalloc((void **)&d_Tile_dnsrow_Val, (denserow_size_1) * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_Tile_dnsrow_Val, matrixA_d->Tile_dnsrow_Val, (denserow_size_1) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

   // char *Tile_dnsrow_idx_1 = (char *)malloc(denserowptr_1[numtileA] * sizeof(char));
    matrixA_d->Tile_dnsrow_idx = (char *)malloc(matrixA_d->denserowptr[matrixA_d->numtile] * sizeof(char));
    memset(matrixA_d->Tile_dnsrow_idx, 0, matrixA_d->denserowptr[matrixA_d->numtile] * sizeof(char));
    char *d_Tile_dnsrow_idx ;
    cudaMalloc((void **)&d_Tile_dnsrow_idx, matrixA_d->denserowptr[matrixA_d->numtile] * sizeof(char));
    cudaMemcpy(d_Tile_dnsrow_idx, matrixA_d->Tile_dnsrow_idx, matrixA_d->denserowptr[matrixA_d->numtile] * sizeof(char), cudaMemcpyHostToDevice);

    //dense column
    //MAT_VAL_TYPE *Tile_dnscol_Val_1=(MAT_VAL_TYPE*)malloc((densecol_size_1) * sizeof(MAT_VAL_TYPE));
    matrixA_d->Tile_dnscol_Val=(MAT_VAL_TYPE*)malloc((densecol_size_1) * sizeof(MAT_VAL_TYPE));
    memset(matrixA_d->Tile_dnscol_Val, 0, (densecol_size_1) * sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_Tile_dnscol_Val;
    cudaMalloc((void **)&d_Tile_dnscol_Val, (densecol_size_1) * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_Tile_dnscol_Val, matrixA_d->Tile_dnscol_Val, (densecol_size_1) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    //char *Tile_dnscol_idx_1 = (char *)malloc(densecolptr_1[numtileA] * sizeof(char));
    matrixA_d->Tile_dnscol_idx = (char *)malloc(matrixA_d->densecolptr[matrixA_d->numtile] * sizeof(char));
    memset(matrixA_d->Tile_dnscol_idx, 0, matrixA_d->densecolptr[matrixA_d->numtile] * sizeof(char));
    char *d_Tile_dnscol_idx;
    cudaMalloc((void **)&d_Tile_dnscol_idx, matrixA_d->densecolptr[matrixA_d->numtile] * sizeof(char));
    cudaMemcpy(d_Tile_dnscol_idx, matrixA_d->Tile_dnscol_idx, matrixA_d->densecolptr[matrixA_d->numtile] * sizeof(char), cudaMemcpyHostToDevice);

//extract COO to a new matrix
    *new_coo_value_temp = (MAT_VAL_TYPE*)malloc(matrixA_d->coocount *sizeof(MAT_VAL_TYPE));
    memset(*new_coo_value_temp, 0, (matrixA_d->coocount) *sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *new_coo_value_1 = *new_coo_value_temp;

    //MAT_VAL_TYPE *new_coo_value_1 = (MAT_VAL_TYPE*)malloc((hybcoosize_1 + coosize_1) *sizeof(MAT_VAL_TYPE));
   // memset(new_coo_value_1, 0, (hybcoosize_1 + coosize_1) *sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_new_coo_value;
    cudaMalloc((void **)&d_new_coo_value, (hybcoosize_1 + coosize_1) *sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_new_coo_value, new_coo_value_1, (hybcoosize_1 + coosize_1) *sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);


    *new_coo_rowidx_temp = (int *)malloc((hybcoosize_1+ coosize_1) *sizeof(int));
    memset(*new_coo_rowidx_temp, 0, (hybcoosize_1+coosize_1) *sizeof(int));
    int *new_coo_rowidx_1 = *new_coo_rowidx_temp;

   // int *new_coo_rowidx_1 = (int *)malloc((hybcoosize_1+ coosize_1) *sizeof(int));
   // memset(new_coo_rowidx_1, 0, (hybcoosize_1+ coosize_1) *sizeof(int));
    int *d_new_coo_rowidx;
    cudaMalloc((void **)&d_new_coo_rowidx, (hybcoosize_1+ coosize_1) *sizeof(int));
    cudaMemcpy(d_new_coo_rowidx, new_coo_rowidx_1, (hybcoosize_1+ coosize_1) *sizeof(int), cudaMemcpyHostToDevice);

    *new_coo_colidx_temp = (int *)malloc((matrixA_d->coocount) *sizeof(int));
    memset(*new_coo_colidx_temp, 0, (matrixA_d->coocount) *sizeof(int));
    int *new_coo_colidx_1 = *new_coo_colidx_temp;

    //int *new_coo_colidx_1 = (int *)malloc((hybcoosize_1+ coosize_1) *sizeof(int));
   // memset(new_coo_colidx_1, 0, (hybcoosize_1+ coosize_1) *sizeof(int));
    int *d_new_coo_colidx;
    cudaMalloc((void **)&d_new_coo_colidx, (hybcoosize_1+ coosize_1) *sizeof(int));
    cudaMemcpy(d_new_coo_colidx, new_coo_colidx_1, (hybcoosize_1+ coosize_1) *sizeof(int), cudaMemcpyHostToDevice);
 
    num_tile_row=1024;
   // unsigned thread;
	unsigned char  *csr_colidx_temp_g_1=(unsigned char*)malloc((num_tile_row * nnz_temp )*sizeof(unsigned char));
    unsigned char  *d_csr_colidx_temp_g;
    cudaMalloc((void **)&d_csr_colidx_temp_g, (num_tile_row * nnz_temp )*sizeof(unsigned char));

    MAT_VAL_TYPE *csr_val_temp_g_1=(MAT_VAL_TYPE*)malloc((num_tile_row * nnz_temp)*sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_csr_val_temp_g;
    cudaMalloc((void **)&d_csr_val_temp_g, (num_tile_row * nnz_temp)*sizeof(MAT_VAL_TYPE));

    int *tile_count_g_1 = (int *)malloc(num_tile_row * tile_count_temp * sizeof(int));
    int *d_tile_count_g;
    cudaMalloc((void **)&d_tile_count_g, num_tile_row * tile_count_temp * sizeof(int));

    MAT_VAL_TYPE *d_value;
    cudaMalloc((void **)&d_value, (matrixA_d->nnz+1) * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_value, matrixA_d->value, (matrixA_d->nnz+1) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
printf("csrptr_offset_1[%d]=%d\n",matrixA_d->numtile,matrixA_d->csrptr_offset[matrixA_d->numtile]);
	cudaMemcpy(d_csrptr_offset, matrixA_d->csrptr_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tile_csr_ptr, matrixA_d->csr_ptr_1, ((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tile_nnz, matrixA_d->tile_nnz, (matrixA_d->numtile+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tile_columnidx, matrixA_d->tile_columnidx, (matrixA_d->numtile+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tile_ptr_A, matrixA_d->tile_ptr, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_csr_offset, matrixA_d->csr_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dns_offset, matrixA_d->dns_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_denserowptr, matrixA_d->denserowptr, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dnsrow_offset, matrixA_d->dnsrow_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_densecolptr, matrixA_d->densecolptr, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dnscol_offset, matrixA_d->dnscol_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ell_offset, matrixA_d->ell_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_coo_offset, matrixA_d->coo_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_blkwidth, matrixA_d->blkwidth, (matrixA_d->numtile + 1)* sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hyb_coocount, matrixA_d->hyb_coocount, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hyb_offset, matrixA_d->hyb_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_blknnz, matrixA_d->blknnz, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_coocount, new_coocount_1, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_csr_colidx_temp_g, 0, (num_tile_row * nnz_temp )*sizeof(unsigned char));
    cudaMemset(d_csr_val_temp_g, 0, (num_tile_row * nnz_temp)*sizeof(MAT_VAL_TYPE));
    cudaMemset(d_tile_count_g, 0, num_tile_row * tile_count_temp * sizeof(int));

    gettimeofday(&t1, NULL);
    for (int blki=0;blki<matrixA_d->tilem;blki+=num_tile_row)//
    {
        //cudaMemset(d_col_flag,0,tilenA*num_tile_row*BLOCK_SIZE * sizeof(unsigned char));
        int start=blki;
        int end= blki+num_tile_row<matrixA_d->tilem ? end= blki+num_tile_row : end=matrixA_d->tilem;
        num_blocks=matrixA_d->tile_ptr[end]-matrixA_d->tile_ptr[start];
       // printf("end=%d  start=%d  num_blocks=%d  blki=%d\n",end,start,num_blocks,blki);
      //  int tilenum_per_row=tile_ptr_A[blki+1]-tile_ptr_A[blki];
        
  //      int rowlen= blki==tilemA-1 ? rowA-(tilemA-1)*BLOCK_SIZE : BLOCK_SIZE ;
        
      //  cudaMemset(d_col_flag,0,tilenA*num_tile_row*BLOCK_SIZE * sizeof(unsigned char));
       // int num_row= blki+num_tile_row<tilemA ? num_row=num_tile_row : num_row=end-start+1;
        cudaMemset(d_tile_count_g, 0, num_tile_row * tile_count_temp * sizeof(int));
        cuda_step4_kernel_1<<<end-start, 32 >>>(matrixA_d->m, matrixA_d->n,d_rowpointerA, d_columnindexA, d_value,
                 matrixA_d->tilem, matrixA_d->tilen, matrixA_d->numtile, d_tile_ptr_A, d_tile_columnidx, d_tile_nnz, d_Format, 
                 d_blknnz, d_tile_csr_ptr, nnz_temp, tile_count_temp,
                 d_csr_colidx_temp_g,d_csr_val_temp_g,d_tile_count_g,blki,num_tile_row);
cudaDeviceSynchronize();

        cuda_step4_kernel_2<<<num_blocks, 32 >>>(matrixA_d->m, matrixA_d->n,d_rowpointerA, d_columnindexA, d_value,
                 matrixA_d->tilem, matrixA_d->tilen, matrixA_d->numtile, d_tile_ptr_A, d_tile_columnidx, d_tile_nnz, d_Format, 
                 d_blknnz, d_tile_csr_ptr, nnz_temp, tile_count_temp,
                 d_csr_colidx_temp_g,d_csr_val_temp_g,d_tile_count_g,
                 d_Tile_csr_Val, d_Tile_csr_Col, d_Tile_csr_Ptr, d_csr_offset, d_csrptr_offset,
                 d_Tile_coo_Val, d_Tile_coo_colIdx, d_Tile_coo_rowIdx, d_coo_offset,
                 d_Tile_ell_Val, d_Tile_ell_colIdx, d_blkwidth, d_ell_offset,
                 d_Tile_hyb_Val, d_Tile_hyb_ellcolIdx, d_Tile_hyb_coorowIdx,  d_hyb_coocount, d_hyb_offset,
                 d_Tile_dns_Val, d_dns_offset,
                 d_Tile_dnsrow_Val, d_Tile_dnsrow_idx, d_denserowptr, d_dnsrow_offset,
                 d_Tile_dnscol_Val, d_Tile_dnscol_idx,  d_densecolptr, d_dnscol_offset,blki,num_tile_row,
                 d_new_coocount,d_new_coo_value, d_new_coo_colidx, d_new_coo_rowidx); 
cudaDeviceSynchronize();

    }
    gettimeofday(&t2, NULL);
    cudaDeviceSynchronize();
    double cuda_time_step4  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("cuda_transform_step4 runtime    = %4.5f ms\n", cuda_time_step4);
    
    cudaMemcpy(matrixA_d->Tile_csr_Val, d_Tile_csr_Val, (csrsize_1)*sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->Tile_csr_Col, d_Tile_csr_Col, (csrsize_1)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->Tile_csr_Ptr, d_Tile_csr_Ptr, (csrtilecount_1 * BLOCK_SIZE)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  // printf("hhhhhhhhhhhhh1\n");
    cudaMemcpy(matrixA_d->Tile_coo_Val, d_Tile_coo_Val, (coosize_1)*sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->Tile_coo_colIdx, d_Tile_coo_colIdx, (coosize_1)*sizeof(unsigned char), cudaMemcpyDeviceToHost);    
    cudaMemcpy(matrixA_d->Tile_coo_rowIdx, d_Tile_coo_rowIdx, (coosize_1)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
//printf("hhhhhhhhhhhhh2\n");
    cudaMemcpy(matrixA_d->Tile_ell_Val, d_Tile_ell_Val, (ellsize_1)*sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->Tile_ell_colIdx, d_Tile_ell_colIdx, (ellsize_1)*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaMemcpy(matrixA_d->Tile_hyb_Val, d_Tile_hyb_Val, (hybellsize_1+hybcoosize_1)*sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->Tile_hyb_ellcolIdx, d_Tile_hyb_ellcolIdx, (hybellsize_1+hybcoosize_1)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->Tile_hyb_coorowIdx, d_Tile_hyb_coorowIdx, (hybcoosize_1)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(matrixA_d->Tile_dns_Val, d_Tile_dns_Val, (dense_size_1)*sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);

    cudaMemcpy(matrixA_d->Tile_dnsrow_Val, d_Tile_dnsrow_Val, (denserow_size_1) * sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->Tile_dnsrow_idx, d_Tile_dnsrow_idx, matrixA_d->denserowptr[matrixA_d->numtile] * sizeof(char), cudaMemcpyDeviceToHost);

    cudaMemcpy(matrixA_d->Tile_dnscol_Val, d_Tile_dnscol_Val, (densecol_size_1) * sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->Tile_dnscol_idx, d_Tile_dnscol_idx, matrixA_d->densecolptr[matrixA_d->numtile] * sizeof(char), cudaMemcpyDeviceToHost);

//printf("hhhhhhhhhhhhh3\n");
	cudaMemcpy(new_coo_value_1, d_new_coo_value, (hybcoosize_1 + coosize_1) *sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
	cudaMemcpy(new_coo_rowidx_1, d_new_coo_rowidx, (hybcoosize_1+coosize_1) *sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(new_coo_colidx_1, d_new_coo_colidx, (hybcoosize_1 + coosize_1) *sizeof(int), cudaMemcpyDeviceToHost);
    //COO new
    printf("hybcoosize + coosize=%d\n",(hybcoosize_1+ coosize_1));
    
    

//printf("hybcoosize + coosize=%d\n",(hybcoosize+ coosize));
 /*   for(int i=0;i<hybcoosize_1 + coosize_1;i++)
    {
        if(new_coo_value[i]!=new_coo_value_1[i])
        {
            printf("step4-error-new_coo_value! i=%d   %f!=%f\n",i,new_coo_value[i],new_coo_value_1[i]);
            break;
        }
    }
    for(int i=0;i<hybcoosize_1 + coosize_1;i++)
    {
        if(new_coo_rowidx[i]!=new_coo_rowidx_1[i])
        {
            printf("step4-error-new_coo_rowidx! i=%d   %d!=%d\n",i,new_coo_rowidx[i],new_coo_rowidx_1[i]);
            break;
        }
    }
    for(int i=0;i<hybcoosize_1 + coosize_1;i++)
    {
        if(new_coo_colidx[i]!=new_coo_colidx_1[i])
        {
            printf("step4-error-new_coo_colidx! i=%d   %d!=%d\n",i,new_coo_colidx[i],new_coo_colidx_1[i]);
            break;
        }
    }*/
    //CSR
  /*  printf("csrsize_1=%d\n",csrsize_1);
    for(int i=0;i<csrsize_1;i++)
    {
        if(matrixA->Tile_csr_Val[i]!=matrixA_d->Tile_csr_Val[i])
        {
            printf("step4-error-Tile_csr_Val! i=%d   %f!=%f\n",i,matrixA->Tile_csr_Val[i],matrixA_d->Tile_csr_Val[i]);
            break;
        }
    }
    for(int i=0;i<csrsize_1;i++)
    {
        if(matrixA->Tile_csr_Col[i]!=matrixA_d->Tile_csr_Col[i])
        {
            printf("step4-error-Tile_csr_Col! i=%d   %d!=%d\n",i,matrixA->Tile_csr_Col[i],matrixA_d->Tile_csr_Col[i]);
            break;
        }
    }
    for(int i=0;i<csrtilecount_1 * BLOCK_SIZE;i++)
    {
        if(matrixA->Tile_csr_Ptr[i]!=matrixA_d->Tile_csr_Ptr[i])
        {
            printf("step4-error-Tile_csr_Ptr! i=%d   %d!=%d\n",i,matrixA->Tile_csr_Ptr[i],matrixA_d->Tile_csr_Ptr[i]);
            break;
        }
    }
    //COO

    //ELL
    printf("ellsize_1=%d\n",ellsize_1);
    for(int i=0;i<ellsize_1;i++)
    {
        if(matrixA->Tile_ell_Val[i]!=matrixA_d->Tile_ell_Val[i])
        {
            printf("step4-error-Tile_ell_Val! i=%d   %f!=%f\n",i,matrixA->Tile_ell_Val[i],matrixA_d->Tile_ell_Val[i]);
            break;
        }
    }
    for(int i=0;i<ellsize_1;i++)
    {
        if(matrixA->Tile_ell_colIdx[i]!=matrixA_d->Tile_ell_colIdx[i])
        {
            printf("step4-error-Tile_ell_colIdx! i=%d   %d!=%d\n",i,matrixA->Tile_ell_colIdx[i],matrixA_d->Tile_ell_colIdx[i]);
            break;
        }
    }
    //HYB
    printf("hybcoosize_1=%d  hybellsize_1+hybcoosize_1=%d\n",hybcoosize_1,hybellsize_1+hybcoosize_1);
    for(int i=0;i<hybellsize_1+hybcoosize_1;i++)
    {
        if(matrixA->Tile_hyb_Val[i]!=matrixA_d->Tile_hyb_Val[i])
        {
            printf("step4-error-Tile_hyb_Val! i=%d   %f!=%f\n",i,matrixA->Tile_hyb_Val[i],matrixA_d->Tile_hyb_Val[i]);
            break;
        }
    }
    for(int i=0;i<hybellsize_1+hybcoosize_1;i++)
    {
        if(matrixA->Tile_hyb_ellcolIdx[i]!=matrixA_d->Tile_hyb_ellcolIdx[i])
        {
            printf("step4-error-Tile_hyb_ellcolIdx! i=%d   %d!=%d\n",i,matrixA->Tile_hyb_ellcolIdx[i],matrixA_d->Tile_hyb_ellcolIdx[i]);
            break;
        }
    }
    for(int i=0;i<hybcoosize_1;i++)
    {
        if(matrixA->Tile_hyb_coorowIdx[i]!=matrixA_d->Tile_hyb_coorowIdx[i])
        {
            printf("step4-error-Tile_hyb_coorowIdx! i=%d   %d!=%d\n",i,matrixA->Tile_hyb_coorowIdx[i],matrixA_d->Tile_hyb_coorowIdx[i]);
            break;
        }
    }
    //dense
    printf("dense_size_1=%d\n",dense_size_1);
    for(int i=0;i<dense_size_1;i++)
    {
        if(matrixA->Tile_dns_Val[i]!=matrixA_d->Tile_dns_Val[i])
        {
            printf("step4-error-Tile_dns_Val! i=%d   %f!=%f\n",i,matrixA->Tile_dns_Val[i],matrixA_d->Tile_dns_Val[i]);
            break;
        }
    }
    //dense row
    printf("denserow_size_1=%d  denserowptr_1[numtileA]=%d\n",denserow_size_1,matrixA_d->denserowptr[matrixA_d->numtile]);
    for(int i=0;i<denserow_size_1;i++)
    {
        if(matrixA->Tile_dnsrow_Val[i]!=matrixA_d->Tile_dnsrow_Val[i])
        {
            printf("step4-error-Tile_dns_Val! i=%d   %f!=%f\n",i,matrixA->Tile_dns_Val[i],matrixA_d->Tile_dnsrow_Val[i]);
            break;
        }
    }//printf("hhhhh=%d\n");
    for(int i=0;i<matrixA_d->denserowptr[matrixA_d->numtile];i++)
    {//printf("i=%d\n",i);
        if(matrixA->Tile_dnsrow_idx[i]!=matrixA_d->Tile_dnsrow_idx[i])
        {
            printf("step4-error-Tile_dnsrow_idx! i=%d   %d!=%d\n",i,matrixA->Tile_dnsrow_idx[i],matrixA_d->Tile_dnsrow_idx[i]);
            break;
        }
    }
    //dense column
    printf("densecol_size_1=%d  densecolptr_1[numtileA]=%d\n",densecol_size_1,matrixA_d->densecolptr[matrixA_d->numtile]);
    for(int i=0;i<densecol_size_1;i++)
    {
        if(matrixA->Tile_dnscol_Val[i]!=matrixA_d->Tile_dnscol_Val[i])
        {
            printf("step4-error-Tile_dnscol_Val! i=%d   %f!=%f\n",i,matrixA->Tile_dnscol_Val[i],matrixA_d->Tile_dnscol_Val[i]);
            break;
        }
    }
    for(int i=0;i<matrixA_d->densecolptr[matrixA_d->numtile];i++)
    {
        if(matrixA->Tile_dnscol_idx[i]!=matrixA_d->Tile_dnscol_idx[i])
        {
            printf("step4-error-Tile_dnscol_idx! i=%d   %d!=%d\n",i,matrixA->Tile_dnscol_idx[i],matrixA_d->Tile_dnscol_idx[i]);
            break;
        }
    }*/
    
    matrixA_d->time_cuda_step=time_cuda_step1+time_cuda_step2+cuda_time_step3+cuda_time_step4;

    
    cudaFree(d_rowpointerA);
    cudaFree(d_columnindexA);
    cudaFree(d_tile_ptr_A);
    cudaFree(d_flag_t);
    cudaFree(d_tile_columnidx);
    cudaFree(d_tile_nnz);
    cudaFree(d_tile_csr_ptr);
    cudaFree(d_j_col);
    cudaFree(d_j_num_t_1);
    cudaFree(d_Format);
    cudaFree(d_blknnz);
    cudaFree(d_dns_offset);
    cudaFree(d_denserowptr);
    cudaFree(d_dnsrow_offset);
    cudaFree(d_densecolptr);
    cudaFree(d_dnscol_offset);
    cudaFree(d_csr_offset);
    cudaFree(d_csrptr_offset);
    cudaFree(d_ell_offset);
    cudaFree(d_coo_offset);
    cudaFree(d_blkwidth);
    cudaFree(d_hyb_coocount);
    cudaFree(d_hyb_offset);
    cudaFree(d_col_flag);
    cudaFree(d_value);
    cudaFree(d_Tile_csr_Val);
    cudaFree(d_Tile_csr_Col);
    cudaFree(d_Tile_csr_Ptr);
    cudaFree(d_Tile_coo_Val);
    cudaFree(d_Tile_coo_colIdx);
    cudaFree(d_Tile_coo_rowIdx);
    cudaFree(d_Tile_ell_Val);
    cudaFree(d_Tile_ell_colIdx);
    cudaFree(d_Tile_hyb_Val);
    cudaFree(d_Tile_hyb_ellcolIdx);
    cudaFree(d_Tile_hyb_coorowIdx);
    cudaFree(d_Tile_dns_Val);
    cudaFree(d_Tile_dnsrow_Val);
    cudaFree(d_Tile_dnsrow_idx);
    cudaFree(d_Tile_dnscol_Val);
    cudaFree(d_Tile_dnscol_idx);
    cudaFree(d_csr_colidx_temp_g);
    cudaFree(d_csr_val_temp_g);
    cudaFree(d_tile_count_g);


}




