void em_cilk_seed_components${'_'+'_'.join(param_val_list)} (
                             int num_components, 
                             int num_dimensions, 
                             int num_events ) 
{
  seed_components${'_'+'_'.join(param_val_list)}(fcs_data_by_event, &components, num_dimensions, num_components, num_events);
}
