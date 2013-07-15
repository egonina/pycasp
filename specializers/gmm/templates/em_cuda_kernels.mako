<%
tempbuff_type_name = 'unsigned int' if supports_float32_atomic_add == '0' else 'float'
%>

#include <stdio.h>

// SPEAKER DIARIZATION GMM TRAINING
__device__ void invert${'_'+'_'.join(param_val_list)}(float* data,
                                                      int actualsize,
                                                      float* log_determinant) {
  int maxsize = actualsize;
  int n = actualsize;
  int num_threads = blockDim.x;
  int num_dimensions = actualsize;
  int row, col;

%if cvtype == 'diag':
  if(threadIdx.x == 0) {
    *log_determinant = 0.0f;
    for(int d = 0; d < num_dimensions*num_dimensions; d++) {
      row = (d) / num_dimensions;
      col = (d) % num_dimensions;
      if(row == col) { //diag only
        *log_determinant += logf(data[row*num_dimensions+col]);
        data[row*num_dimensions+col] = 1.0f / data[row*num_dimensions+col];
      }
    }
  }

%else:
  if(threadIdx.x == 0) {
    *log_determinant = 0.0f;
    
    // sanity check
    if (actualsize == 1) {
      *log_determinant = logf(data[0]);
      data[0] = 1.0 / data[0];
    } else {

      for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
      for (int i=1; i < actualsize; i++) {
        for (int j=i; j < actualsize; j++) { // do a column of L
          float sum = 0.0f;
          for (int k = 0; k < i; k++)
            sum += data[j*maxsize+k] * data[k*maxsize+i];
          data[j*maxsize+i] -= sum;
        }
        if (i == actualsize-1) continue;
        for (int j=i+1; j < actualsize; j++) { // do a row of U
          float sum = 0.0f;
          for (int k = 0; k < i; k++)
            sum += data[i*maxsize+k]*data[k*maxsize+j];
          data[i*maxsize+j] =
            (data[i*maxsize+j]-sum) / data[i*maxsize+i];
        }
      }

      for(int i=0; i<actualsize; i++) {
        *log_determinant += logf(fabs(data[i*n+i]));
      }

      for ( int i = 0; i < actualsize; i++ ) // invert L
        for ( int j = i; j < actualsize; j++ ) {
          float x = 1.0f;
          if ( i != j ) {
            x = 0.0f;
            for ( int k = i; k < j; k++ )
              x -= data[j*maxsize+k]*data[k*maxsize+i];
          }
          data[j*maxsize+i] = x / data[j*maxsize+j];
        }
      for ( int i = 0; i < actualsize; i++ ) // invert U
        for ( int j = i; j < actualsize; j++ ) {
          if ( i == j ) continue;
          float sum = 0.0f;
          for ( int k = i; k < j; k++ )
            sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
          data[i*maxsize+j] = -sum;
        }
      for ( int i = 0; i < actualsize; i++ ) // final inversion
        for ( int j = 0; j < actualsize; j++ ) {
          float sum = 0.0f;
          for ( int k = ((i>j)?i:j); k < actualsize; k++ )
            sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
          data[j*maxsize+i] = sum;
        }
    }

  }
%endif
}
                                                                                
__device__ void seed_covars${'_'+'_'.join(param_val_list)}(components_t* components,
                                                           float* fcs_data,
                                                           float* means,
                                                           int num_dimensions,
                                                           int num_events,
                                                           float* avgvar,
                                                           int num_components) {
    // access thread id
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int row, col;

    // Compute average variance for each dimension
    for(int i = tid; i < num_dimensions * num_dimensions; i+= num_threads) {
      // Add the average variance divided by a constant, this keeps the cov matrix from becoming singular
      row = (i) / num_dimensions;
      col = (i) % num_dimensions;

      components->R[row*num_dimensions + col] = 0.0f;
      for(int j=0; j < num_events; j++) {
        if(row==col) {
          components->R[row*num_dimensions + col] +=
              (fcs_data[j*num_dimensions + row]) * (fcs_data[j*num_dimensions + row]);
        }
      }
      if(row==col) {
        components->R[row*num_dimensions+col] /= (float)(num_events - 1);
        components->R[row*num_dimensions+col] -=
            ((float)(num_events) * means[row] * means[row]) /
            (float)(num_events-1);
        components->R[row*num_dimensions+col] /= (float)num_components;
      }
    }
}

__device__ void average_variance${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                                                float* means,
                                                                int num_dimensions,
                                                                int num_events,
                                                                float* avgvar) {
    // access thread id
    int tid = threadIdx.x;
    
    __shared__ float variances[${max_num_dimensions}];
    __shared__ float total_variance;
    
    // Compute average variance for each dimension
    if(tid < num_dimensions) {
        variances[tid] = 0.0f;
        // Sum up all the variance
        for(int j=0; j < num_events; j++) {
            // variance = (data - mean)^2
          variances[tid] += (fcs_data[j*num_dimensions + tid]) *
                            (fcs_data[j*num_dimensions + tid]);
        }
        variances[tid] /= (float) num_events;
        variances[tid] -= means[tid] * means[tid];
    }
    
    __syncthreads();
    
    if(tid == 0) {
        total_variance = 0.0f;
        for(int i = 0; i < num_dimensions; i++) {
          total_variance += variances[i];
        }
        *avgvar = total_variance / (float) num_dimensions;
    }
}

__device__ void compute_constants${'_'+'_'.join(param_val_list)}(components_t* components,
                                                                 int num_components,
                                                                 int num_dimensions) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    //int num_elements = ${max_num_dimensions}*${max_num_dimensions};
    int num_elements = num_dimensions * num_dimensions;
    int row, col;

    __shared__ float determinant_arg; // only one thread computes the inverse so we need a shared argument
    
    float log_determinant = 0.0f;
    
    __shared__ float matrix[${max_num_dimensions} * ${max_num_dimensions}];
    
    // Invert the matrix for every component
    int c = blockIdx.x;

    // Copy the R matrix into shared memory for doing the matrix inversion
    for(int i = tid; i < num_elements; i+= num_threads ) {
        matrix[i] = components->R[c * num_dimensions * num_dimensions + i];
    }
      
    __syncthreads(); 

    invert${'_'+'_'.join(param_val_list)}(matrix,
                                          num_dimensions,
                                          &determinant_arg);

    __syncthreads(); 

    log_determinant = determinant_arg;

    __syncthreads();
    
    // Copy the matrx from shared memory back into the component memory
    for(int i = tid; i < num_elements; i += num_threads) {
        components->Rinv[c * num_dimensions * num_dimensions + i] = matrix[i];
    }
    
    __syncthreads();
    
    // Compute the constant
    // Equivilent to: log(1/((2*PI)^(M/2)*det(R)^(1/2)))
    // This constant is used in all E-step likelihood calculations
    if(tid == 0) {
      components->constant[c] = -num_dimensions * 0.5 *
                                logf(2 * PI) - 0.5 * log_determinant;
      components->CP[c] = components->constant[c] * 2;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! @param fcs_data         FCS data: [num_events]
//! @param components         Clusters: [num_components]
//! @param num_dimensions   number of dimensions in an FCS event
//! @param num_events       number of FCS events
////////////////////////////////////////////////////////////////////////////////

__global__ void
seed_components${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                               components_t* components,
                                               int num_dimensions,
                                               int num_components,
                                               int num_events) {
    // access thread id
    int tid = threadIdx.x;
    // access number of threads in this block
    int num_threads = blockDim.x;

    // shared memory
    __shared__ float means[${max_num_dimensions}];
    
    // Compute the means
    mvtmeans(fcs_data, num_dimensions, num_events, means);
   
    __syncthreads();
    
    __shared__ float avgvar;
    
    // Seed first covariance matrix
    //Compute the average variance
    seed_covars${'_'+'_'.join(param_val_list)}(components,
                                               fcs_data,
                                               means,
                                               num_dimensions,
                                               num_events,
                                               &avgvar,
                                               num_components);
    average_variance${'_'+'_'.join(param_val_list)}(fcs_data,
                                                    means,
                                                    num_dimensions,
                                                    num_events,
                                                    &avgvar);    
    int num_elements;
    int row, col;
        
    // Number of elements in the covariance matrix
    num_elements = num_dimensions*num_dimensions;

     __syncthreads();

    float seed;
    if(num_components > 1) {
       seed = (num_events)/(num_components);
    } else {
        seed = 0.0f;
    }

    if(tid < num_dimensions) {
      components->means[tid] = means[tid];
    }

    __syncthreads();
    
    // Seed the pi, means, and covariances for every component
    for(int c = 1; c < num_components; c++) {
        if(tid < num_dimensions) {
            components->means[c*num_dimensions+tid] =
                fcs_data[((int)(c*seed))*num_dimensions+tid];
       
          }
        //Seed the rest of the covariance matrices
        for(int i = tid; i < num_elements; i+= num_threads) {
          components->R[c*num_dimensions*num_dimensions+i] = components->R[i];
          components->Rinv[c*num_dimensions*num_dimensions+i] = 0.0f;
        }
    }

    //compute pi, N
    for(int c = 0; c < num_components; c++) {
        if(tid == 0) {
          components->pi[c] = 1.0f/((float)num_components);
          components->N[c] = ((float) num_events) / ((float)num_components);
          components->avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;
        }
    }
}

__global__ void compute_average_variance${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                                                        components_t* components,
                                                                        int num_dimensions,
                                                                        int num_components,
                                                                        int num_events) {
    // access thread id
    int tid = threadIdx.x;
    // access number of threads in this block
    int num_threads = blockDim.x;

    // shared memory
    __shared__ float means[${max_num_dimensions}];
    
    // Compute the means
    mvtmeans(fcs_data, num_dimensions, num_events, means);
   
    __syncthreads();
    
    __shared__ float avgvar;
    
    average_variance${'_'+'_'.join(param_val_list)}(fcs_data,
                                                    means,
                                                    num_dimensions,
                                                    num_events,
                                                    &avgvar);    
    __syncthreads();
    
    for(int c = 0; c < num_components; c++) {
      if(tid == 0) {
        components->avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;
      }
    }
}


__device__ void compute_indices${'_'+'_'.join(param_val_list)}(int num_events,
                                                               int* start,
                                                               int* stop) {
    // Break up the events evenly between the blocks
    int num_pixels_per_block = num_events / ${num_blocks_estep};
    // Make sure the events being accessed by the block are aligned to a multiple of 16
    num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);

    *start = blockIdx.x * num_pixels_per_block + threadIdx.x;
    
    // Last block will handle the leftover events
    if(blockIdx.x == ${num_blocks_estep}-1) {
        *stop = num_events;
    } else {
        *stop = (blockIdx.x+1) * num_pixels_per_block;
    }
}

__global__ void
estep1${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                      components_t* components,
                                      float* component_memberships,
                                      int num_dimensions,
                                      int num_events,
                                      float* loglikelihoods) {
    
    // Cached component parameters
  __shared__ float means[${max_num_dimensions}];
  __shared__ float Rinv[${max_num_dimensions}*${max_num_dimensions}];
    float component_pi;
    float constant;
    const unsigned int tid = threadIdx.x;
 
    int start_index;
    int end_index;

    int c = blockIdx.y;
    int num_components = gridDim.y;

    compute_indices${'_'+'_'.join(param_val_list)}(num_events,
                                                   &start_index,
                                                   &end_index);
    __syncthreads();
    
    float like;

    // This loop computes the expectation of every event into every component
    //
    // P(k|n) = L(x_n|mu_k,R_k)*P(k) / P(x_n)
    //
    // Compute log-likelihood for every component for each event
    // L = constant*exp(-0.5*(x-mu)*Rinv*(x-mu))
    // log_L = log_constant - 0.5*(x-u)*Rinv*(x-mu)
    // the constant stored in components[c].constant is already the log of the constant
    
    // copy the means for this component into shared memory
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }

    // copy the covariance inverse into shared memory
    for(int i = tid; i < num_dimensions*num_dimensions; i+= ${num_threads_estep}) {
        Rinv[i] = components->Rinv[c*num_dimensions*num_dimensions+i];
    }

    component_pi = components->pi[c];
    constant = components->constant[c];
        
    // Sync to wait for all params to be loaded to shared memory
    __syncthreads();
    
    if(blockIdx.y == 0) {
      for(int event=start_index; event<end_index; event += ${num_threads_estep}) {
        loglikelihoods[event] = 0.0f; 
      }
    }

    __syncthreads();

    
    for(int event=start_index; event<end_index; event += ${num_threads_estep}) {
        like = 0.0f;
%if cvtype == 'diag':
        for(int j=0; j<num_dimensions; j++) {
            like += (fcs_data[j*num_events+event] - means[j]) *
                (fcs_data[j*num_events+event] - means[j]) * Rinv[j*num_dimensions+j];
        }

%else:
        for(int i=0; i<num_dimensions; i++) {
            for(int j=0; j<num_dimensions; j++) {
                like += (fcs_data[i*num_events+event] - means[i]) *
                    (fcs_data[j*num_events+event] - means[j]) * Rinv[i*num_dimensions+j];
            }
        }
%endif
        component_memberships[c*num_events+event] =
            (component_pi > 0.0f) ? -0.5 * like + constant + logf(component_pi) :
            MINVALUEFORMINUSLOG;

    }
}

__global__ void
estep1_log_add${'_'+'_'.join(param_val_list)}(int num_events,
                                              int num_components,
                                              float* loglikelihoods,
                                              float* component_memberships) {
    int tid = threadIdx.x;
    for(int event = tid; event < num_events; event += ${num_threads_estep}) {
      float log_lkld = MINVALUEFORMINUSLOG;
      for(int c = 0; c<num_components; c++) {
        log_lkld = log_add(log_lkld,
                           component_memberships[c*num_events+event]);
      }
      loglikelihoods[event] = log_lkld;
    }
}
    
__global__ void
estep2${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                      components_t* components,
                                      float* component_memberships,
                                      int num_dimensions,
                                      int num_components,
                                      int num_events,
                                      float* likelihood) {
    float temp;
    float thread_likelihood = 0.0f;
    __shared__ float total_likelihoods[${num_threads_estep}];
    float max_likelihood;
    float denominator_sum;
    
    // Break up the events evenly between the blocks
    int num_pixels_per_block = num_events / ${num_blocks_estep};
    // Make sure the events being accessed by the block are aligned to a multiple of 16
    num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);
    int tid = threadIdx.x;
    
    int start_index;
    int end_index;
    start_index = blockIdx.x * num_pixels_per_block + tid;
    
    // Last block will handle the leftover events
    if(blockIdx.x == ${num_blocks_estep}-1) {
        end_index = num_events;
    } else {
        end_index = (blockIdx.x+1) * num_pixels_per_block;
    }

    __syncthreads();

    total_likelihoods[tid] = 0.0f;

    // P(x_n) = sum of likelihoods weighted by P(k) (their probability, cluster[c].pi)
    // However we use logs to prevent under/overflow
    //  log-sum-exp formula:
    //  log(sum(exp(x_i)) = max(z) + log(sum(exp(z_i-max(z))))
    for(int pixel = start_index; pixel < end_index; pixel += ${num_threads_estep}) {
      // find the maximum likelihood for this event
      max_likelihood = component_memberships[pixel];
      for(int c = 1; c < num_components; c++) {
        max_likelihood = fmaxf(max_likelihood,
                               component_memberships[c*num_events+pixel]);
      }
      
      // Compute P(x_n), the denominator of the probability (sum of weighted likelihoods)
      denominator_sum = 0.0f;
      for(int c = 0; c < num_components; c++) {
        temp = expf(component_memberships[c*num_events+pixel] - max_likelihood);
        denominator_sum += temp;
      }
      temp = max_likelihood + logf(denominator_sum);
      thread_likelihood += temp;
      //thread_likelihood += loglikelihoods[pixel];
        
      // Divide by denominator, also effectively normalize probabilities
      for(int c = 0; c < num_components; c++) {
        component_memberships[c*num_events+pixel] =
            expf(component_memberships[c*num_events+pixel] - temp);
      }
      
    }

    __syncthreads();
    
    total_likelihoods[tid] = thread_likelihood;
    
    __syncthreads();

    parallelSum(total_likelihoods);
    if(tid == 0) {
      likelihood[blockIdx.x] = total_likelihoods[0];
    }
}

__global__ void
mstep_means${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                           components_t* components,
                                           float* component_memberships,
                                           int num_dimensions,
                                           int num_components,
                                           int num_events) {
    // One block per component, per dimension:  (M x D) grid of blocks
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int c = blockIdx.x; // component number
    int d = blockIdx.y; // dimension number
    
    __shared__ float temp_sum[${num_threads_mstep}];
    float sum = 0.0f;
    
    for(int event = tid; event < num_events; event+= num_threads) {
      sum += fcs_data[d*num_events+event] *
             component_memberships[c*num_events+event];
    }
    temp_sum[tid] = sum;
    
    __syncthreads();

    if(tid == 0) {
      for(int i=1; i < num_threads; i++) {
        temp_sum[0] += temp_sum[i];
      }
      components->means[c*num_dimensions+d] =
          temp_sum[0] / components->N[c];
    }
}

__global__ void
mstep_means_idx${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                               int* indices,
                                               int num_indices,
                                               components_t* components,
                                               float* component_memberships,
                                               int num_dimensions,
                                               int num_components,
                                               int num_events) {
    // One block per component, per dimension:  (M x D) grid of blocks
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int c = blockIdx.x; // component number
    int d = blockIdx.y; // dimension number

    __shared__ float temp_sum[${num_threads_mstep}];
    float sum = 0.0f;

    int event;
    
    for(int index = tid; index < num_indices; index += num_threads) {
      event = indices[index];
      sum += fcs_data[d*num_events+event] *
             component_memberships[c*num_events+event];
    }
    temp_sum[tid] = sum;
    __syncthreads();
    
    parallelSum(temp_sum);
    if(tid == 0) {
        components->means[c*num_dimensions+d] =
            temp_sum[0] / components->N[c];
    }
}

__global__ void
mstep_means_transpose${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                                     components_t* components,
                                                     float* component_memberships,
                                                     int num_dimensions,
                                                     int num_components,
                                                     int num_events) {
    // One block per component, per dimension:  (M x D) grid of blocks
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int c = blockIdx.y; // component number
    int d = blockIdx.x; // dimension number

    __shared__ float temp_sum[${num_threads_mstep}];
    float sum = 0.0f;
    
    for(int event = tid; event < num_events; event += num_threads) {
        sum += fcs_data[d*num_events+event] *
               component_memberships[c*num_events+event];
    }
    temp_sum[tid] = sum;
    
    __syncthreads();

    if(tid == 0) {
      for(int i = 1; i < num_threads; i++) {
        temp_sum[0] += temp_sum[i];
      }
      components->means[c*num_dimensions+d] =
          temp_sum[0] / components->N[c];
    }
}

__global__ void
mstep_N${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                       components_t* components,
                                       float* component_memberships,
                                       int num_dimensions,
                                       int num_components,
                                       int num_events) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int c = blockIdx.x;

    __shared__ float avgvar;

    // Need to store the sum computed by each thread so in the end
    // a single thread can reduce to get the final sum
    __shared__ float temp_sums[${num_threads_mstep}];

    // Compute new N
    float sum = 0.0f;
    // Break all the events accross the threads, add up probabilities
    for(int event=tid; event < num_events; event += num_threads) {
        sum += component_memberships[c*num_events+event];
    }
    temp_sums[tid] = sum;
 
    __syncthreads();

    // Let the first thread add up all the intermediate sums
    // Could do a parallel reduction...doubt it's really worth it for so few elements though
    if(tid == 0) {
      components->N[c] = 0.0f;
      for(int j=0; j<num_threads; j++) {
        components->N[c] += temp_sums[j];
      }

      // Set PI to the # of expected items, and then normalize it later
      components->pi[c] = components->N[c];
    }
}

__global__ void
mstep_N_idx${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                           int* indices,
                                           int num_indices,
                                           components_t* components,
                                           float* component_memberships,
                                           int num_dimensions,
                                           int num_components,
                                           int num_events) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int c = blockIdx.x;
    
    // Need to store the sum computed by each thread so in the end
    // a single thread can reduce to get the final sum
    __shared__ float temp_sums[${num_threads_mstep}];

    // Compute new N
    float sum = 0.0f;
    // Break all the events accross the threads, add up probabilities

    int event;
    for(int index = tid; index < num_indices; index += num_threads) {
      event = indices[index];
      sum += component_memberships[c*num_events+event];
    }
    temp_sums[tid] = sum;
 
    __syncthreads();
    
    parallelSum(temp_sums);
    if(tid == 0) {
        components->N[c] = temp_sums[0];
        components->pi[c] = temp_sums[0];
    }
}
 
%if covar_version_name.upper() in ['1','V1','_V1']:

__global__ void
mstep_covariance${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                                components_t* components,
                                                float* component_memberships,
                                                int num_dimensions,
                                                int num_components,
                                                int num_events) {
    int tid = threadIdx.x; // easier variable name for our thread ID
    int row,col,c;
    compute_row_col(num_dimensions, &row, &col);
    c = blockIdx.x; // Determines what component this block is handling    

    int matrix_index = row * num_dimensions + col;

%if cvtype == 'diag':
    if(row!=col) {
      components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
      matrix_index = col*num_dimensions+row;
      components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
      return;
    }
%endif
    // Store the means of this component in shared memory
    __shared__ float means[${max_num_dimensions}];
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }
    __syncthreads();

    __shared__ float temp_sums[${num_threads_mstep}];
    
    float cov_sum = 0.0f;
    for(int event = tid; event < num_events; event+=${num_threads_mstep}) {
      cov_sum += (fcs_data[row*num_events+event] - means[row]) *
                 (fcs_data[col*num_events+event] - means[col]) *
                 component_memberships[c*num_events+event];
    }
    temp_sums[tid] = cov_sum;

    __syncthreads();
    
    parallelSum(temp_sums);
    if(tid == 0) {
      cov_sum = temp_sums[0];
      if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
        cov_sum /= components->N[c];
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
      } else {
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
      }
      if(row == col) {
        components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
      }
    }
}

void mstep_covar_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                       float* d_fcs_data_by_event,
                                                       components_t* d_components,
                                                       float* d_component_memberships,
                                                       int num_dimensions,
                                                       int num_components,
                                                       int num_events,
                                                       ${tempbuff_type_name}* temp_buffer_2b) {
  dim3 gridDim2(num_components,num_dimensions*(num_dimensions+1)/2);
  mstep_covariance${'_'+'_'.join(param_val_list)}
      <<<gridDim2, ${num_threads_mstep}>>>(d_fcs_data_by_dimension,
                                           d_components,
                                           d_component_memberships,
                                           num_dimensions,
                                           num_components,
                                           num_events);
}

__global__ void
mstep_covariance_idx${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                                    int* indices,
                                                    int num_indices,
                                                    components_t* components,
                                                    float* component_memberships,
                                                    int num_dimensions,
                                                    int num_components,
                                                    int num_events) {
    int tid = threadIdx.x; // easier variable name for our thread ID
    int row,col,c;
    compute_row_col(num_dimensions, &row, &col);
    c = blockIdx.x; // Determines what component this block is handling    

    int matrix_index = row * num_dimensions + col;

    // Store the means of this component in shared memory
    __shared__ float means[${max_num_dimensions}];
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }
    __syncthreads();

    __shared__ float temp_sums[${num_threads_mstep}];
    
    float cov_sum = 0.0f;

    int event = 0;
    
    for(int index = tid; index < num_indices; index+=${num_threads_mstep}) {
      event = indices[index];
    
      cov_sum += (fcs_data[row*num_events+event] - means[row]) *
                 (fcs_data[col*num_events+event] - means[col]) *
                 component_memberships[c*num_events+event];
 
    }
    temp_sums[tid] = cov_sum;
    
    __syncthreads();

    parallelSum(temp_sums);
    if(tid == 0) {
      cov_sum = temp_sums[0];
      if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
        cov_sum /= components->N[c];
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
      } else {
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
      }
      if(row == col) {
        components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
      }
    }
}

void mstep_covar_launch_idx${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                           float* d_fcs_data_by_event,
                                                           int* d_indices,
                                                           int num_indices,
                                                           components_t* d_components,
                                                           float* d_component_memberships,
                                                           int num_dimensions,
                                                           int num_components,
                                                           int num_events,
                                                           ${tempbuff_type_name}* temp_buffer_2b) {
  dim3 gridDim2(num_components,num_dimensions*(num_dimensions+1)/2);
  mstep_covariance_idx${'_'+'_'.join(param_val_list)}
      <<<gridDim2, ${num_threads_mstep}>>>(d_fcs_data_by_dimension,
                                           d_indices,
                                           num_indices,
                                           d_components,
                                           d_component_memberships,
                                           num_dimensions,
                                           num_components,
                                           num_events);
}

%elif covar_version_name.upper() in ['2','2A','V2','V2A','_V2','_V2A']:

__global__ void
mstep_covariance${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                                components_t* components,
                                                float* component_memberships,
                                                int num_dimensions,
                                                int num_components,
                                                int num_events) {
    int tid = threadIdx.x; // easier variable name for our thread ID
    int row,col,c;
    compute_row_col_thread(num_dimensions, &row, &col);

    __syncthreads();
    
    c = blockIdx.x; // Determines what component this block is handling    

    int matrix_index = row * num_dimensions + col;
    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[${max_num_dimensions}];
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }
    __syncthreads();

    float cov_sum = 0.0f; //my local sum for the matrix element, I (thread) sum up over all N events into this var

    if(tid < num_dimensions*(num_dimensions+1)/2) {
%if cvtype == 'diag':
      if(row!=col) {
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        return;      
      }
%endif

        for(int event=0; event < num_events; event++) {
          cov_sum += (fcs_data[event*num_dimensions+row] - means[row]) *
                     (fcs_data[event*num_dimensions+col] - means[col]) *
                     component_memberships[c*num_events+event];
        }
    }

    __syncthreads();

    if(tid < num_dimensions*(num_dimensions+1)/2) {

        if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
          cov_sum /= components->N[c];
          components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
          // Set the symmetric value
          matrix_index = col*num_dimensions+row;
          components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        } else {
          components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
          // Set the symmetric value
          matrix_index = col*num_dimensions+row;
          components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
        }
        
        if(row == col) {
          components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
        }
    }   
}

void mstep_covar_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                       float* d_fcs_data_by_event,
                                                       components_t* d_components,
                                                       float* d_component_memberships,
                                                       int num_dimensions,
                                                       int num_components,
                                                       int num_events,
                                                       ${tempbuff_type_name}* temp_buffer_2b) {
  int num_threads = num_dimensions*(num_dimensions+1)/2;
  mstep_covariance${'_'+'_'.join(param_val_list)}
      <<<num_components, num_threads>>>(d_fcs_data_by_event,
                                        d_components,
                                        d_component_memberships,
                                        num_dimensions,
                                        num_components,
                                        num_events);
}

__global__ void
mstep_covariance_idx${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                                    int* indices,
                                                    int num_indices,
                                                    components_t* components,
                                                    float* component_memberships,
                                                    int num_dimensions,
                                                    int num_components,
                                                    int num_events) {
    int tid = threadIdx.x; // easier variable name for our thread ID
    int row,col,c;
    compute_row_col_thread(num_dimensions, &row, &col);

    __syncthreads();
    
    c = blockIdx.x; // Determines what component this block is handling    

    int matrix_index = row * num_dimensions + col;
    
    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[${max_num_dimensions}];
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }
    __syncthreads();

    float cov_sum = 0.0f; //my local sum for the matrix element, I (thread) sum up over all N events into this var
    for(int index=0; index < num_indices; index++) {
      int event = indices[index];
      cov_sum += (fcs_data[event*num_dimensions+row] - means[row]) *
                 (fcs_data[event*num_dimensions+col] - means[col]) *
                 component_memberships[c*num_events+event];      
    }

    __syncthreads();

    if(tid < num_dimensions*(num_dimensions+1)/2) {
        if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
          cov_sum /= components->N[c];
          components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
          // Set the symmetric value
          matrix_index = col*num_dimensions+row;
          components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        } else {
          components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
          // Set the symmetric value
          matrix_index = col*num_dimensions+row;
          components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        }
        
        if(row == col) {
          components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
        }
    }   
}

void mstep_covar_launch_idx${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                           float* d_fcs_data_by_event,
                                                           int* d_indices,
                                                           int num_indices,
                                                           components_t* d_components,
                                                           float* d_component_memberships,
                                                           int num_dimensions,
                                                           int num_components,
                                                           int num_events,
                                                           ${tempbuff_type_name}* temp_buffer_2b) {
  int num_threads = num_dimensions*(num_dimensions+1)/2;
  mstep_covariance_idx${'_'+'_'.join(param_val_list)}
      <<<num_components, num_threads>>>(d_fcs_data_by_event,
                                        d_indices,
                                        num_indices,
                                        d_components,
                                        d_component_memberships,
                                        num_dimensions,
                                        num_components,
                                        num_events);
}

%elif covar_version_name.upper() in ['2B','V2B','_V2B']:
/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a M x B blocks and D x D/2 threads:
 * B is the number of event blocks (N/events_per_block)
 */

__global__ void
mstep_covariance${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                                components_t* components,
                                                float* component_memberships,
                                                int num_dimensions,
                                                int num_components,
                                                int num_events,
                                                int event_block_size,
                                                int num_b,
                                                ${tempbuff_type_name}* temp_buffer) {
  //mstep_covariance${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float* component_memberships, int num_dimensions, int num_components, int num_events, int event_block_size, int num_b, float *temp_buffer) {

  int tid = threadIdx.x; // easier variable name for our thread ID
    
    // Determine what row,col this matrix is handling, also handles the symmetric element
    int row,col,c;
    compute_row_col_thread(num_dimensions, &row, &col);

    int e_start, e_end;
    compute_my_event_indices(num_events, event_block_size, num_b, &e_start, &e_end);
    c = blockIdx.x; // Determines what component this block is handling    
    int matrix_index = row * num_dimensions + col;

    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[${max_num_dimensions}];
    __shared__ float myR[${max_num_dimensions}*${max_num_dimensions}];
    
    // copy the means for this component into shared memory
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }
    __syncthreads();
    
    //my local sum for the matrix element,
    //I (thread) sum up over all N events into this var
    float cov_sum = 0.0f; 

    if(tid < num_dimensions*(num_dimensions+1)/2) {

%if cvtype == 'diag':
      if(row!=col) {
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        return;      
      }
%endif
        for(int event=e_start; event < e_end; event++) {
          cov_sum += (fcs_data[event*num_dimensions+row] - means[row]) *
                     (fcs_data[event*num_dimensions+col] - means[col]) *
                     component_memberships[c*num_events+event];
        }

        myR[matrix_index] = cov_sum;
     
%if supports_float32_atomic_add != '0':
        float old = atomicAdd(&(temp_buffer[c*num_dimensions*num_dimensions+matrix_index]),
                              myR[matrix_index]); 
%else:
        int old = atomicAdd(&(temp_buffer[c*num_dimensions*num_dimensions+matrix_index]),
                            ToFixedPoint(myR[matrix_index])); 
%endif
    }
    __syncthreads();

    if(tid < num_dimensions*(num_dimensions+1)/2) {

      if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?

%if supports_float32_atomic_add != '0':
        float cs = temp_buffer[c*num_dimensions*num_dimensions+matrix_index];
%else:
        float cs = ToFloatPoint(temp_buffer[c*num_dimensions*num_dimensions+matrix_index]);
%endif
        cs /= components->N[c];
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cs;
        // Set the symmetric value
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cs;
      } else {
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        // Set the symmetric value
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
      }
    
      if(row == col) {
        components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
      }
    }
}
 
void mstep_covar_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                       float* d_fcs_data_by_event,
                                                       components_t* d_components,
                                                       float* d_component_memberships,
                                                       int num_dimensions,
                                                       int num_components,
                                                       int num_events,
                                                       ${tempbuff_type_name}* temp_buffer_2b) {
  int num_event_blocks = ${num_event_blocks};
  int event_block_size = num_events%${num_event_blocks} == 0 ? num_events /
       ${num_event_blocks}:num_events/(${num_event_blocks}-1);
  dim3 gridDim2(num_components,num_event_blocks);
  int num_threads = num_dimensions*(num_dimensions+1)/2;
  mstep_covariance${'_'+'_'.join(param_val_list)}
      <<<gridDim2, num_threads>>>(d_fcs_data_by_event,
                                  d_components,
                                  d_component_memberships,
                                  num_dimensions,
                                  num_components,
                                  num_events,
                                  event_block_size,
                                  num_event_blocks,
                                  temp_buffer_2b);
}

__global__ void
mstep_covariance_idx${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                                    int* indices,
                                                    int num_indices,
                                                    components_t* components,
                                                    float* component_memberships,
                                                    int num_dimensions,
                                                    int num_components,
                                                    int num_events,
                                                    int event_block_size,
                                                    int num_b,
                                                    ${tempbuff_type_name}* temp_buffer) {
    int tid = threadIdx.x; // easier variable name for our thread ID
    
    // Determine what row,col this matrix is handling, also handles the symmetric element
    int row,col,c;
    compute_row_col_thread(num_dimensions, &row, &col);

    //int e_start, e_end;
    //compute_my_event_indices(num_events, event_block_size, num_b, &e_start, &e_end);

    int i_start, i_end;              
    compute_my_event_indices(num_indices, event_block_size, num_b, &i_start, &i_end);

    c = blockIdx.x; // Determines what component this block is handling    
    int matrix_index = row * num_dimensions + col;

    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[${max_num_dimensions}];
    __shared__ float myR[${max_num_dimensions}*${max_num_dimensions}];
    
    // copy the means for this component into shared memory
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }
    __syncthreads();
    //my local sum for the matrix element,
    //I (thread) sum up over all N events into this var
    float cov_sum = 0.0f; 

    if(tid < num_dimensions*(num_dimensions+1)/2) {
        int event;
        for(int index=i_start; index < i_end; index++) {
          event = indices[index];
          cov_sum += (fcs_data[event*num_dimensions+row] - means[row]) *
                     (fcs_data[event*num_dimensions+col] - means[col]) *
                     component_memberships[c*num_events+event];
        }

        myR[matrix_index] = cov_sum;
     
%if supports_32b_floating_point_atomics != '0':
        float old = atomicAdd(&(temp_buffer[c*num_dimensions*num_dimensions+matrix_index]),
                              myR[matrix_index]); 
%else:
        int old = atomicAdd(&(temp_buffer[c*num_dimensions*num_dimensions+matrix_index]), 
                            ToFixedPoint(myR[matrix_index]));
%endif
    }
    __syncthreads();

    if(tid < num_dimensions*(num_dimensions+1)/2) {
      if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
%if supports_32b_floating_point_atomics != '0':
        float cs = temp_buffer[c*num_dimensions*num_dimensions+matrix_index];
%else:
        float cs = ToFloatPoint(temp_buffer[c*num_dimensions*num_dimensions+matrix_index]);
%endif
        cs /= components->N[c];
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cs;
        // Set the symmetric value
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cs;
      } else {
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        // Set the symmetric value
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
      }
    
      if(row == col) {
        components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
      }
    }
}

void mstep_covar_launch_idx${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                           float* d_fcs_data_by_event,
                                                           int* d_indices,
                                                           int num_indices,
                                                           components_t* d_components,
                                                           float* d_component_memberships,
                                                           int num_dimensions,
                                                           int num_components,
                                                           int num_events,
                                                           ${tempbuff_type_name}* temp_buffer_2b) {
  int num_event_blocks = ${num_event_blocks};
  int event_block_size = num_indices%${num_event_blocks} == 0 ? num_indices /
       ${num_event_blocks}:num_indices/(${num_event_blocks}-1);
  dim3 gridDim2(num_components,num_event_blocks);
  int num_threads = num_dimensions*(num_dimensions+1)/2;
  mstep_covariance_idx${'_'+'_'.join(param_val_list)}
      <<<gridDim2, num_threads>>>(d_fcs_data_by_event,
                                  d_indices,
                                  num_indices,
                                  d_components,
                                  d_component_memberships,
                                  num_dimensions,
                                  num_components,
                                  num_events,
                                  event_block_size,
                                  num_event_blocks,
                                  temp_buffer_2b);
}

%else:

/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a D*D/2 blocks: 
 */
__global__ void
mstep_covariance${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float* component_memberships, int num_dimensions, int num_components, int num_events) {
  int tid = threadIdx.x; // easier variable name for our thread ID
  int row,col;
  compute_row_col_block(num_dimensions, &row, &col);
  int matrix_index = row * num_dimensions + col;

  // Store ALL the means in shared memory
  __shared__ float means[${max_num_components_covar_v3}*${max_num_dimensions_covar_v3}];
  for(int i = tid; i<num_components*num_dimensions; i+=${num_threads_mstep}) {
    means[i] = components->means[i];
  }
  __syncthreads();

  __shared__ float temp_sums[${num_threads_mstep}];
  //local storage for component results
  __shared__ float component_sum[${max_num_components_covar_v3}];
  
  for(int c = 0; c<num_components; c++) {

%if cvtype == 'diag':
    if(row!=col) {
      components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
      matrix_index = col*num_dimensions+row;
      components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
      return;
    }
%endif    
    float cov_sum = 0.0f;
    for(int event=tid; event < num_events; event+=${num_threads_mstep}) {
      cov_sum += (fcs_data[row*num_events+event] - means[c*num_dimensions+row]) *
                 (fcs_data[col*num_events+event] - means[c*num_dimensions+col]) *
                 component_memberships[c*num_events+event];
    }
    temp_sums[tid] = cov_sum;
    
    __syncthreads();
    
    parallelSum(temp_sums);
    if(tid == 0) {
      component_sum[c] = temp_sums[0];
    }
    // __syncthreads();
  }
  __syncthreads();
    
  for(int c = tid; c<num_components; c+=${num_threads_mstep}) {
    matrix_index =  row * num_dimensions + col;
    if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
      component_sum[c] /= components->N[c];

      components->R[c*num_dimensions*num_dimensions+matrix_index] = component_sum[c];
      matrix_index = col*num_dimensions+row;
      components->R[c*num_dimensions*num_dimensions+matrix_index] = component_sum[c];
    } else {
      components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
      matrix_index = col*num_dimensions+row;
      components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
    }
    if(row == col) {
      components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
    }
  } 
}

void mstep_covar_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                       float* d_fcs_data_by_event,
                                                       components_t* d_components,
                                                       float* d_component_memberships,
                                                       int num_dimensions,
                                                       int num_components,
                                                       int num_events,
                                                       ${tempbuff_type_name}* temp_buffer_2b) {
  int num_blocks = num_dimensions*(num_dimensions+1)/2;
  mstep_covariance${'_'+'_'.join(param_val_list)}
      <<<num_blocks, ${num_threads_mstep}>>>(d_fcs_data_by_dimension,
                                             d_components,
                                             d_component_memberships,
                                             num_dimensions,
                                             num_components,
                                             num_events);
}

__global__ void
mstep_covariance_idx${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                                    int* indices,
                                                    int num_indices,
                                                    components_t* components,
                                                    float* component_memberships,
                                                    int num_dimensions,
                                                    int num_components,
                                                    int num_events) {
  int tid = threadIdx.x; // easier variable name for our thread ID
  int row,col;
  compute_row_col_block(num_dimensions, &row, &col);
      
  int matrix_index;
    
  // Store ALL the means in shared memory
  __shared__ float means[${max_num_components_covar_v3}*${max_num_dimensions_covar_v3}];
  for(int i = tid; i<num_components*num_dimensions; i+=${num_threads_mstep}) {
    means[i] = components->means[i];
  }
  __syncthreads();

  __shared__ float temp_sums[${num_threads_mstep}];
  //local storage for component results
  __shared__ float component_sum[${max_num_components_covar_v3}];
  
  for(int c = 0; c<num_components; c++) {
    float cov_sum = 0.0f;
    int event;
    for(int index=tid; index < num_indices; index+=${num_threads_mstep}) {
      event = indices[index];
      cov_sum += (fcs_data[row*num_events+event] - means[c*num_dimensions+row]) *
                 (fcs_data[col*num_events+event] - means[c*num_dimensions+col]) *
                 component_memberships[c*num_events+event];
    }
    temp_sums[tid] = cov_sum;
    
    __syncthreads();
      
    parallelSum(temp_sums);
    if(tid == 0) {
      component_sum[c] = temp_sums[0]; 
    }
    __syncthreads();
  }
  __syncthreads();
    
  for(int c = tid; c<num_components; c+=${num_threads_mstep}) {
    matrix_index =  row * num_dimensions + col;
    if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
      component_sum[c] /= components->N[c];
      components->R[c*num_dimensions*num_dimensions+matrix_index] = component_sum[c];
      matrix_index = col*num_dimensions+row;
      components->R[c*num_dimensions*num_dimensions+matrix_index] = component_sum[c];
    } else {
      components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
      matrix_index = col*num_dimensions+row;
      components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
    }
    if(row == col) {
      components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
    }
  } 
}

void mstep_covar_launch_idx${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                           float* d_fcs_data_by_event,
                                                           int* d_index_list,
                                                           int num_indices,
                                                           components_t* d_components,
                                                           float* d_component_memberships,
                                                           int num_dimensions,
                                                           int num_components,
                                                           int num_events,
                                                           ${tempbuff_type_name}* temp_buffer_2b) {
  int num_blocks = num_dimensions*(num_dimensions+1)/2;
  mstep_covariance_idx${'_'+'_'.join(param_val_list)}
      <<<num_blocks, ${num_threads_mstep}>>>(d_fcs_data_by_dimension,
                                             d_index_list,
                                             num_indices,
                                             d_components,
                                             d_component_memberships,
                                             num_dimensions,
                                             num_components,
                                             num_events);
}

%endif

/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a M x D*D grid of blocks: 
 *  i.e. dim3 gridDim(num_components,num_dimensions*num_dimensions)
 */
__global__ void
mstep_covariance_transpose${'_'+'_'.join(param_val_list)}(float* fcs_data,
                                                          components_t* components,
                                                          float* component_memberships,
                                                          int num_dimensions,
                                                          int num_components,
                                                          int num_events) {
    int tid = threadIdx.x; // easier variable name for our thread ID

    // Determine what row,col this matrix is handling, also handles the symmetric element
    int row,col,c;
    compute_row_col_transpose(num_dimensions, &row, &col);

    __syncthreads();
    
    c = blockIdx.y; // Determines what component this block is handling    

    int matrix_index = row * num_dimensions + col;

%if cvtype == 'diag':
    if(row != col) {
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        return;
    }
%endif 
    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[${max_num_dimensions}];
    // copy the means for this component into shared memory
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }

    // Sync to wait for all params to be loaded to shared memory
    __syncthreads();

    __shared__ float temp_sums[${num_threads_mstep}];
    
    float cov_sum = 0.0f;

    for(int event=tid; event < num_events; event+=${num_threads_mstep}) {
        cov_sum += (fcs_data[row*num_events+event] - means[row]) *
                   (fcs_data[col*num_events+event] - means[col]) *
                   component_memberships[c*num_events+event]; 
    }
    temp_sums[tid] = cov_sum;

    __syncthreads();
    
    parallelSum(temp_sums);
    if(tid == 0) {
        cov_sum = temp_sums[0];
        if(components->N[c] >= 1.0f) { 
            cov_sum /= components->N[c];
            components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
            matrix_index = col*num_dimensions+row;
            components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        } else {
            components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
            matrix_index = col*num_dimensions+row;
            components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
        }
        if(row == col) {
            components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
        }
    }
}

void seed_components_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_event,
                                                           components_t* d_components,
                                                           int num_dimensions,
                                                           int num_components,
                                                           int num_events) {
  seed_components${'_'+'_'.join(param_val_list)}
      <<< 1, ${num_threads_estep}>>>(d_fcs_data_by_event,
                                     d_components,
                                     num_dimensions,
                                     num_components,
                                     num_events);
}

void compute_average_variance_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_event,
                                                                    components_t* d_components,
                                                                    int num_dimensions,
                                                                    int num_components,
                                                                    int num_events) {
  compute_average_variance${'_'+'_'.join(param_val_list)}
      <<< 1, ${num_threads_estep}>>>(d_fcs_data_by_event,
                                     d_components,
                                     num_dimensions,
                                     num_components,
                                     num_events);
  cudaThreadSynchronize();
}

void estep1_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                  components_t* d_components,
                                                  float* d_component_memberships,
                                                  int num_dimensions,
                                                  int num_components,
                                                  int num_events,
                                                  float* d_loglikelihoods) {
  estep1${'_'+'_'.join(param_val_list)}
      <<<dim3(${num_blocks_estep},num_components), ${num_threads_estep}>>>(d_fcs_data_by_dimension,
                                                                           d_components,
                                                                           d_component_memberships,
                                                                           num_dimensions,
                                                                           num_events,
                                                                           d_loglikelihoods);

  //new step to log add the component log probabilities
  estep1_log_add${'_'+'_'.join(param_val_list)}
      <<<1, ${num_threads_estep}>>>(num_events,
                                    num_components,
                                    d_loglikelihoods,
                                    d_component_memberships);
}

void estep2_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                  components_t* d_components,
                                                  float* d_component_memberships,
                                                  int num_dimensions,
                                                  int num_components,
                                                  int num_events,
                                                  float* d_likelihoods) {
  estep2${'_'+'_'.join(param_val_list)}
     <<<${num_blocks_estep}, ${num_threads_estep}>>>(d_fcs_data_by_dimension,
                                                     d_components,
                                                     d_component_memberships,
                                                     num_dimensions,
                                                     num_components,
                                                     num_events,
                                                     d_likelihoods);
}

void mstep_N_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_event,
                                                   components_t* d_components,
                                                   float* d_component_memberships,
                                                   int num_dimensions,
                                                   int num_components,
                                                   int num_events) {
  mstep_N${'_'+'_'.join(param_val_list)}
     <<<num_components, ${num_threads_mstep}>>>(d_fcs_data_by_event,
                                                d_components,
                                                d_component_memberships,
                                                num_dimensions,
                                                num_components,
                                                num_events);
}

void mstep_N_launch_idx${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_event,
                                                       int* d_index_list,
                                                       int num_indices,
                                                       components_t* d_components,
                                                       float* d_component_memberships,
                                                       int num_dimensions,
                                                       int num_components,
                                                       int num_events) {
  mstep_N_idx${'_'+'_'.join(param_val_list)}
     <<<num_components, ${num_threads_mstep}>>>(d_fcs_data_by_event,
                                                d_index_list,
                                                num_indices,
                                                d_components,
                                                d_component_memberships,
                                                num_dimensions,
                                                num_components,
                                                num_events);
}

void mstep_means_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                       components_t* d_components,
                                                       float* d_component_memberships,
                                                       int num_dimensions,
                                                       int num_components,
                                                       int num_events) {
  dim3 gridDim1(num_components,num_dimensions);
  mstep_means${'_'+'_'.join(param_val_list)}
     <<<gridDim1, ${num_threads_mstep}>>>(d_fcs_data_by_dimension,
                                          d_components,
                                          d_component_memberships,
                                          num_dimensions,
                                          num_components,
                                          num_events);
}

void mstep_means_launch_idx${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension,
                                                           int* indices,
                                                           int num_indices,
                                                           components_t* d_components,
                                                           float* d_component_memberships,
                                                           int num_dimensions,
                                                           int num_components,
                                                           int num_events) {
  dim3 gridDim1(num_components,num_dimensions);
  mstep_means_idx${'_'+'_'.join(param_val_list)}
      <<<gridDim1, ${num_threads_mstep}>>>(d_fcs_data_by_dimension, 
                                           indices,
                                           num_indices,
                                           d_components,
                                           d_component_memberships,
                                           num_dimensions,
                                           num_components,
                                           num_events);
}

/*
 * Computes the constant for each component and normalizes pi for every component
 * In the process it inverts R and finds the determinant
 * 
 * Needs to be launched with the number of blocks = number of components
 */
__global__ void
constants_kernel${'_'+'_'.join(param_val_list)}(components_t* components,
                                                int num_components,
                                                int num_dimensions) {
    compute_constants${'_'+'_'.join(param_val_list)}(components,
                                                     num_components,
                                                     num_dimensions);
    
    __syncthreads();
    
     if(blockIdx.x == 0) {
       normalize_pi(components,num_components);
    }
}

void constants_kernel_launch${'_'+'_'.join(param_val_list)}(components_t* d_components,
                                                            int num_components,
                                                            int num_dimensions) {
  constants_kernel${'_'+'_'.join(param_val_list)}
     <<<num_components, ${num_threads_mstep}>>>(d_components,
                                                num_components,
                                                num_dimensions);
}
