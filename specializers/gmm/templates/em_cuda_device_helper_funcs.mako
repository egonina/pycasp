#define PI  3.1415926535897931
#define COVARIANCE_DYNAMIC_RANGE 1E6
#include <stdio.h>
#include <float.h>
#include <limits.h>
#define MINVALUEFORMINUSLOG -1000.0f
#define MIN_VARIANCE 0.01
#define SCALE_VAL 1000.0f


__device__ static int  ToFixedPoint(float input) {
        if (input==FLT_MAX)
                return INT_MAX;
        return (int)(SCALE_VAL*input);
}

__device__ static float ToFloatPoint(int input) {
        if (input==INT_MAX)
                return FLT_MAX;
        return (float)input/SCALE_VAL;
}

__device__ float log_add(float log_a, float log_b) {
  if(log_a < log_b) {
      float tmp = log_a;
      log_a = log_b;
      log_b = tmp;
    }
  //setting MIN...LOG so small, I don't even need to look
  return (((log_b - log_a) <= MINVALUEFORMINUSLOG) ? log_a : 
                log_a + (float)(logf(1.0 + (double)(expf((double)(log_b - log_a))))));
}



/*
 * Compute the multivariate mean of the FCS data
 */ 
__device__ void mvtmeans(float* fcs_data, int num_dimensions, int num_events, float* means) {
    // access thread id
    int tid = threadIdx.x;

    if(tid < num_dimensions) {
        means[tid] = 0.0f;

        // Sum up all the values for the dimension
        for(int i=0; i < num_events; i++) {
            means[tid] += fcs_data[i*num_dimensions+tid];
        }

        // Divide by the # of elements to get the average
        means[tid] /= (float) num_events;
    }
}


__device__ void normalize_pi(components_t* components, int num_components) {
    __shared__ float sum;
    
    // TODO: could maybe use a parallel reduction..but the # of elements is really small
    // What is better: having thread 0 compute a shared sum and sync, or just have each one compute the sum?
    if(threadIdx.x == 0) {
        sum = 0.0f;
        for(int i=0; i<num_components; i++) {
            sum += components->pi[i];
        }
    }
    
    __syncthreads();
    
    for(int c=threadIdx.x; c < num_components; c += blockDim.x) {
        components->pi[threadIdx.x] /= sum;
    }
 
    __syncthreads();
}

__device__ void parallelSum(float* data) {
  const unsigned int tid = threadIdx.x;
  for(unsigned int s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s)
      data[tid] += data[tid + s];  
    __syncthreads();
  }
  if (tid < 32) {
    volatile float* sdata = data;
    sdata[tid] += sdata[tid+32];
    sdata[tid] += sdata[tid+16];
    sdata[tid] += sdata[tid+8];
    sdata[tid] += sdata[tid+4];
    sdata[tid] += sdata[tid+2];
    sdata[tid] += sdata[tid+1];
  }
}

/*
 * Computes the row and col of a square matrix based on the index into
 * a lower triangular (with diagonal) matrix
 * 
 * Used to determine what row/col should be computed for covariance
 * based on a block index.
 */
__device__ void compute_row_col(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == blockIdx.y) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}

//CODEVAR_2
__device__ void compute_row_col_thread(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == threadIdx.x) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}
//CODEVAR_3
__device__ void compute_row_col_block(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == blockIdx.x) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}

//CODEVAR_2B and CODEVAR_3B
__device__ void compute_my_event_indices(int n, int bsize, int num_b, int* e_start, int* e_end) {
  int myId = blockIdx.y;
  *e_start = myId*bsize;
  if(myId==(num_b-1)) {
    *e_end = ((myId*bsize)-n < 0 ? n:myId*bsize);
  } else {
    *e_end = myId*bsize + bsize;
  }
  
  return;
}


__device__ void compute_row_col_transpose(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == blockIdx.x) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}

