void em_cuda_seed_components${'_'+'_'.join(param_val_list)} (
                             int num_components, 
                             int num_dimensions, 
                             int num_events ) 
{
  seed_components_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_event, d_components, num_dimensions, num_components, num_events);
  cudaThreadSynchronize();
}
