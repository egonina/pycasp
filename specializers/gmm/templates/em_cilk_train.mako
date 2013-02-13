
boost::python::tuple em_cilk_train${'_'+'_'.join(param_val_list)} (
                             int num_components, 
                             int num_dimensions, 
                             int num_events,
                             int min_iters,
                             int max_iters) 
{
    
    // Computes the R matrix inverses, and the gaussian constant
    constants${'_'+'_'.join(param_val_list)}(&components,num_components,num_dimensions);
    // Compute average variance based on the data
    compute_average_variance${'_'+'_'.join(param_val_list)}(fcs_data_by_event, &components, num_dimensions, num_components, num_events);

    // Calculate an epsilon value
    //int ndata_points = num_events*num_dimensions;
    float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)num_events*num_dimensions)*0.0001;
    int iters;
    float likelihood = -100000;
    float old_likelihood = likelihood * 10;
    
    float change = epsilon*2;
    
    iters = 0;
    // This is the iterative loop for the EM algorithm.
    // It re-estimates parameters, re-computes constants, and then regroups the events
    // These steps keep repeating until the change in likelihood is less than some epsilon        
    while(iters < min_iters || (fabs(change) > epsilon && iters < max_iters)) {
        old_likelihood = likelihood;

        estep1${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events,loglikelihoods);
        estep2${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events,&likelihood);
        
        // This kernel computes a new N, pi isn't updated until compute_constants though
        mstep_n${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events);
        mstep_mean${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events);
        mstep_covar${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events);

        
        // Inverts the R matrices, computes the constant, normalizes cluster probabilities
        constants${'_'+'_'.join(param_val_list)}(&components,num_components,num_dimensions);
        change = likelihood - old_likelihood;
        iters++;
    }

    estep1${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events,loglikelihoods);
    estep2${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events,&likelihood);
    
  return boost::python::make_tuple(likelihood, iters);
}
