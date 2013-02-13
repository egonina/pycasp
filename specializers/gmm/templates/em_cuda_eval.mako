void em_cuda_eval${'_'+'_'.join(param_val_list)} (
                             int num_components, 
                             int num_dimensions, 
                             int num_events ) 
{
  //TODO: Is this necessary, or can we assume the values are still set?
  // Computes the R matrix inverses, and the gaussian constant
  constants_kernel_launch${'_'+'_'.join(param_val_list)}(d_components,num_components,num_dimensions);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Constants Kernel execution failed: ");
  estep1_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_components, d_component_memberships, num_dimensions,num_components,num_events,d_loglikelihoods);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Kernel execution failed");

  copy_evals_data_GPU_to_CPU(num_events, num_components);
}

