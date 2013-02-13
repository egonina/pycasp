
void seed_components${'_'+'_'.join(param_val_list)}(float *data, components_t* components, int D, int M, int N);
void constants${'_'+'_'.join(param_val_list)}(components_t* components, int M, int D);
void estep1${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N, float* loglikelihoods);
void estep2${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N, float* likelihood);
void mstep_n${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N);
void mstep_n_idx${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_event, int* d_index_list, int num_indices,components_t* d_components, float* component_memberships, int num_dimensions, int num_components, int num_events);
void mstep_mean${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N);
void mstep_mean_idx${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, int* d_index_list, int num_indices, components_t* d_components, float* component_memberships, int num_dimensions, int num_components, int num_events);
void mstep_covar${'_'+'_'.join(param_val_list)}(float* data, components_t* components,float* component_memberships, int D, int M, int N);
void mstep_covar_idx${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event,int* d_index_list, int num_indices, components_t* d_components, float* component_memberships, int num_dimensions, int num_components, int num_events);
