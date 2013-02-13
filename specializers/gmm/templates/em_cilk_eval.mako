void em_cilk_eval${'_'+'_'.join(param_val_list)} (
                             int num_components, 
                             int num_dimensions, 
                             int num_events) 
{
  //TODO: Is this necessary, or can we assume the values are still set?
  // Computes the R matrix inverses, and the gaussian constant
  constants${'_'+'_'.join(param_val_list)}(&components,num_components,num_dimensions);
  estep1${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events,loglikelihoods);
}
