<%
tempbuff_type_name = 'unsigned int' if supports_float32_atomic_add == '0' else 'float'
%>

boost::python::tuple em_cuda_train${'_'+'_'.join(param_val_list)} (
                             int num_components, 
                             int num_dimensions, 
                             int num_events,
                             int min_iters,
                             int max_iters) 
{

  // ================= Temp buffer for codevar 2b ================ 
  ${tempbuff_type_name} *temp_buffer_2b = NULL;
%if covar_version_name.upper() in ['2B','V2B','_V2B']:
    //scratch space to clear out components->R
    ${tempbuff_type_name} *zeroR_2b = (${tempbuff_type_name}*) malloc(sizeof(${tempbuff_type_name})*num_dimensions*num_dimensions*num_components);
    for(int i = 0; i<num_dimensions*num_dimensions*num_components; i++) {
        zeroR_2b[i] = 0;
    }
    CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_buffer_2b),sizeof(${tempbuff_type_name})*num_dimensions*num_dimensions*num_components));
    CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_2b, zeroR_2b, sizeof(${tempbuff_type_name})*num_dimensions*num_dimensions*num_components, cudaMemcpyHostToDevice) );
%endif

  // Computes the R matrix inverses, and the gaussian constant
  constants_kernel_launch${'_'+'_'.join(param_val_list)}(d_components,num_components,num_dimensions);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Constants Kernel execution failed: ");
  //copy_component_data_GPU_to_CPU(num_components, num_dimensions);
  //print_components(&components, num_components, num_dimensions);
  //copy_evals_data_GPU_to_CPU(num_events, num_components);
  //print_evals(component_memberships, loglikelihoods, num_events, num_components);

  // Compute average variance based on the data
  compute_average_variance_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_event, d_components, num_dimensions, num_components, num_events);
  
  // Calculate an epsilon value
  float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)num_events*num_dimensions)*0.0001;
  int iters;
  float likelihood = -100000;
  float old_likelihood = likelihood * 10;
  // Used to hold the result from regroup kernel
  float* likelihoods = (float*) malloc(sizeof(float)*${num_blocks_estep});
  float* d_likelihoods;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_likelihoods, sizeof(float)*${num_blocks_estep}));
    
  /*************** EM ALGORITHM *****************************/
        
  //================================== EM INITIALIZE =======================


  float change = epsilon*2;

  //================================= EM BEGIN ==================================
  iters = 0;

  // This is the iterative loop for the EM algorithm.
  // It re-estimates parameters, re-computes constants, and then regroups the events
  // These steps keep repeating until the change in likelihood is less than some epsilon        

  while(iters < min_iters || (iters < max_iters && fabs(change) > epsilon)) {
    old_likelihood = likelihood;

    
    estep1_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_components, d_component_memberships, num_dimensions,num_components,num_events,d_loglikelihoods);
    estep2_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_components, d_component_memberships, num_dimensions,num_components,num_events,d_likelihoods);
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("E-step Kernel execution failed");
    //copy_evals_data_GPU_to_CPU(num_events, num_components);
    //print_evals(component_memberships, loglikelihoods, num_events, num_components);

    // Copy the likelihood totals from each block, sum them up to get a total
    CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*${num_blocks_estep},cudaMemcpyDeviceToHost));
    likelihood = 0.0;
    for(int i=0;i<${num_blocks_estep};i++) {
      likelihood += likelihoods[i]; 
    }
    // This kernel computes a new N, pi isn't updated until compute_constants though
    mstep_N_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_event,d_components, d_component_memberships, num_dimensions,num_components,num_events);
    cudaThreadSynchronize();

    // This kernel computes new means
    mstep_means_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_components, d_component_memberships, num_dimensions,num_components,num_events);
    cudaThreadSynchronize();

            
%if covar_version_name.upper() in ['2B','V2B','_V2B']:
      CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_2b, zeroR_2b, sizeof(${tempbuff_type_name})*num_dimensions*num_dimensions*num_components, cudaMemcpyHostToDevice) );
%endif

    // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per component
    mstep_covar_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_fcs_data_by_event,d_components,d_component_memberships,num_dimensions,num_components,num_events,temp_buffer_2b);
    cudaThreadSynchronize();
                 
    CUT_CHECK_ERROR("M-step Kernel execution failed: ");

    // Inverts the R matrices, computes the constant, normalizes component probabilities
    constants_kernel_launch${'_'+'_'.join(param_val_list)}(d_components,num_components,num_dimensions);
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("Constants Kernel execution failed: ");

    // change = (likelihood - old_likelihood)/fabs(old_likelihood);
     change = likelihood - old_likelihood;

    iters++;
    
    //copy_component_data_GPU_to_CPU(num_components, num_dimensions);
    //print_components(&components, num_components, num_dimensions);
  }//EM Loop

    
    //regroup = E step
    // Compute new component membership probabilities for all the events
  estep1_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_components,d_component_memberships, num_dimensions,num_components,num_events,d_loglikelihoods);
    estep2_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_components,d_component_memberships, num_dimensions,num_components,num_events,d_likelihoods);
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("E-step Kernel execution failed: ");
        
    CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*${num_blocks_estep},cudaMemcpyDeviceToHost));
    likelihood = 0.0;
    for(int i=0;i<${num_blocks_estep};i++) {
      likelihood += likelihoods[i];
    }

  cudaThreadSynchronize();

  //================================ EM DONE ==============================

  copy_component_data_GPU_to_CPU(num_components, num_dimensions);
  copy_evals_data_GPU_to_CPU(num_events, num_components);
  
%if covar_version_name.upper() in ['2B','V2B','_V2B']:
  free(zeroR_2b);
  CUDA_SAFE_CALL(cudaFree(temp_buffer_2b));
%endif

   return boost::python::make_tuple(likelihood, iters);
}


