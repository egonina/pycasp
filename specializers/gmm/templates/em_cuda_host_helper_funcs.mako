
#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
   }

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);            \


//=== Data structure pointers ===

//GPU copies of events
float* d_fcs_data_by_event;
float* d_fcs_data_by_dimension;

//GPU index list for train_on_subset
int* d_index_list;

//GPU copies of components
components_t temp_components;
components_t* d_components;

//GPU copies of eval data
float *d_component_memberships;
float *d_loglikelihoods;

//Copy functions to ensure CPU data structures are up to date
void copy_component_data_GPU_to_CPU(int num_components, int num_dimensions);
void copy_evals_data_GPU_to_CPU(int num_events, int num_components);

//=== Memory Alloc/Free Functions ===

// ================== Event data allocation on GPU  ================= :

void alloc_events_on_GPU(int num_dimensions, int num_events) {
  int mem_size = num_dimensions*num_events*sizeof(float);
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_event, mem_size));
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_dimension, mem_size));
  CUT_CHECK_ERROR("Alloc events on GPU failed: ");
}

void alloc_index_list_on_GPU(int num_indices) {
  int mem_size = num_indices*sizeof(int);
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_index_list, mem_size));
  CUT_CHECK_ERROR("Alloc index list on GPU failed: ");
}

void alloc_events_from_index_on_GPU(int num_indices, int num_dimensions) {
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_fcs_data_by_event, sizeof(float)*num_indices*num_dimensions));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_fcs_data_by_dimension, sizeof(float)*num_indices*num_dimensions));
  CUT_CHECK_ERROR("Alloc events from index on GPU failed: ");
}

// ================== Cluster data allocation on GPU  ================= :
size_t alloc_components_on_GPU(int original_num_components, int num_dimensions) {
  // Setup the component data structures on device
  // First allocate structures on the host, CUDA malloc the arrays
  // Then CUDA malloc structures on the device and copy them over
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.N),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.pi),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.CP),sizeof(float)*original_num_components)); //NEW LINE
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.constant),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.avgvar),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.means),sizeof(float)*num_dimensions*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.R),sizeof(float)*num_dimensions*num_dimensions*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.Rinv),sizeof(float)*num_dimensions*num_dimensions*original_num_components));

  // Allocate a struct on the device
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_components, sizeof(components_t)));

  // Copy Cluster data to device
  CUDA_SAFE_CALL(cudaMemcpy(d_components,&temp_components,sizeof(components_t),cudaMemcpyHostToDevice));
  CUT_CHECK_ERROR("Alloc components on GPU failed: ");

  //return (PyObject*)temp_components.means;
  return (size_t)temp_components.means;
}


// ================= Eval data alloc on GPU =============== 
void alloc_evals_on_GPU(int num_events, int num_components){
  CUDA_SAFE_CALL(cudaMalloc((void**) &(d_component_memberships),sizeof(float)*num_events*num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(d_loglikelihoods),sizeof(float)*num_events));
  CUT_CHECK_ERROR("Alloc eval data on GPU failed: ");
}


// ======================== Copy event data from CPU to GPU ================
void copy_event_data_CPU_to_GPU(int num_events, int num_dimensions) {
  int mem_size = num_dimensions*num_events*sizeof(float);
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_event, fcs_data_by_event, mem_size,cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_dimension, fcs_data_by_dimension, mem_size,cudaMemcpyHostToDevice) );
  CUT_CHECK_ERROR("Copy events from CPU to GPU execution failed: ");
}

void copy_index_list_data_CPU_to_GPU(int num_indices) {
  int mem_size = num_indices*sizeof(int);
  CUDA_SAFE_CALL(cudaMemcpy( d_index_list, index_list, mem_size,cudaMemcpyHostToDevice) );
  CUT_CHECK_ERROR("Copy event index list from CPU to GPU execution failed: ");
}

void copy_events_from_index_CPU_to_GPU(int num_indices, int num_dimensions) {
  CUDA_SAFE_CALL(cudaMemcpy(d_fcs_data_by_dimension, fcs_data_by_dimension, sizeof(float)*num_indices*num_dimensions, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_fcs_data_by_event, fcs_data_by_event, sizeof(float)*num_indices*num_dimensions, cudaMemcpyHostToDevice));
  CUT_CHECK_ERROR("Copy events by index from CPU to GPU execution failed: ");
}

// ======================== Copy component data from CPU to GPU ================
void copy_component_data_CPU_to_GPU(int num_components, int num_dimensions) {
  CUDA_SAFE_CALL(cudaMemcpy(temp_components.N, components.N, sizeof(float)*num_components,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(temp_components.pi, components.pi, sizeof(float)*num_components,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(temp_components.CP, components.CP, sizeof(float)*num_components,cudaMemcpyHostToDevice)); 
  //NEW LINE
  CUDA_SAFE_CALL(cudaMemcpy(temp_components.constant, components.constant, sizeof(float)*num_components,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(temp_components.avgvar, components.avgvar, sizeof(float)*num_components,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(temp_components.means, components.means, sizeof(float)*num_dimensions*num_components,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(temp_components.R, components.R, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(temp_components.Rinv, components.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_components,&temp_components,sizeof(components_t),cudaMemcpyHostToDevice));
  CUT_CHECK_ERROR("Copy components from CPU to GPU execution failed: ");
}


// ======================== Copy component data from GPU to CPU ================
void copy_component_data_GPU_to_CPU(int num_components, int num_dimensions) {
  CUDA_SAFE_CALL(cudaMemcpy(&temp_components, d_components, sizeof(components_t),cudaMemcpyDeviceToHost));
  // copy all of the arrays from the structs
  CUDA_SAFE_CALL(cudaMemcpy(components.N, temp_components.N, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.pi, temp_components.pi, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.CP, temp_components.CP, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.constant, temp_components.constant, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.avgvar, temp_components.avgvar, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.means, temp_components.means, sizeof(float)*num_dimensions*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.R, temp_components.R, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.Rinv, temp_components.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyDeviceToHost));
  CUT_CHECK_ERROR("Copy components from GPU to CPU execution failed: ");
}

// ======================== Copy eval data GPU <==> CPU ================
void copy_evals_CPU_to_GPU(int num_events, int num_components) {
  CUDA_SAFE_CALL(cudaMemcpy( d_loglikelihoods, loglikelihoods, sizeof(float)*num_events,cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(cudaMemcpy( d_component_memberships, component_memberships, sizeof(float)*num_events*num_components,cudaMemcpyHostToDevice) );
   CUT_CHECK_ERROR("Copy eval data from CPU to GPU execution failed: ");
}

void copy_evals_data_GPU_to_CPU(int num_events, int num_components){
  CUDA_SAFE_CALL(cudaMemcpy(component_memberships, d_component_memberships, sizeof(float)*num_events*num_components, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(loglikelihoods, d_loglikelihoods, sizeof(float)*num_events, cudaMemcpyDeviceToHost));
  //  CUDA_SAFE_CALL(cudaMemcpy(likelihoods, d_likelihoods, sizeof(float)*num_events, cudaMemcpyDeviceToHost));
  CUT_CHECK_ERROR("Copy eval data from GPU to CPU execution failed: ");
}

// ================== Event data dellocation on GPU  ================= :
void dealloc_events_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_event));
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_dimension));
  CUT_CHECK_ERROR("Dealloc events on GPU failed: ");
}

void dealloc_index_list_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(d_index_list));
  CUT_CHECK_ERROR("Dealloc index list on GPU failed: ");
}

// ==================== Cluster data deallocation on GPU =================  
void dealloc_components_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(temp_components.N));
  CUDA_SAFE_CALL(cudaFree(temp_components.pi));
  CUDA_SAFE_CALL(cudaFree(temp_components.CP));
  CUDA_SAFE_CALL(cudaFree(temp_components.constant));
  CUDA_SAFE_CALL(cudaFree(temp_components.avgvar));
  CUDA_SAFE_CALL(cudaFree(temp_components.means));
  CUDA_SAFE_CALL(cudaFree(temp_components.R));
  CUDA_SAFE_CALL(cudaFree(temp_components.Rinv));
  CUDA_SAFE_CALL(cudaFree(d_components));
  CUT_CHECK_ERROR("Dealloc components on GPU failed: ");
}

// ==================== Eval data deallocation GPU =================  
void dealloc_evals_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(d_component_memberships));
  CUDA_SAFE_CALL(cudaFree(d_loglikelihoods));
  CUT_CHECK_ERROR("Dealloc eval data on GPU failed: ");
}

// ==================== Diagnostics =================

void print_components(int num_components, int num_dimensions){
  copy_component_data_GPU_to_CPU(num_components,num_dimensions);
  printf("===============\n");
  for(int m = 0; m < num_components; m++){
	printf("%0.4f ", components.N[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
	printf("%0.4f ", components.pi[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
	printf("%0.4f ", components.CP[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
	printf("%0.4f ", components.constant[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
	printf("%0.4f ", components.avgvar[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
    for(int d = 0; d < num_dimensions; d++)
        printf("%0.4f ", components.means[m*num_dimensions+d]);
    printf("\n");
  }
    for(int m = 0; m < num_components; m++){
        for(int d = 0; d < num_dimensions; d++)
            for(int d2 = 0; d2 < num_dimensions; d2++)
                printf("%0.4f ", components.R[m*num_dimensions*num_dimensions+d*num_dimensions+d2]);
        printf("\n");
    }

    for(int m = 0; m < num_components; m++){
        for(int d = 0; d < num_dimensions; d++)
            for(int d2 = 0; d2 < num_dimensions; d2++)
                printf("%0.4f ", components.Rinv[m*num_dimensions*num_dimensions+d*num_dimensions+d2]);
        printf("\n");
    }
  printf("===============\n");
}

