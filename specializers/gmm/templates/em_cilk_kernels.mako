
void seed_covars${'_'+'_'.join(param_val_list)}(components_t* components, float* fcs_data, float* means, int num_dimensions, int num_events, float* avgvar, int num_components) {

    cilk_for(int i=0; i < num_dimensions*num_dimensions; i++) {
      int row = (i) / num_dimensions;
      int col = (i) % num_dimensions;
      components->R[row*num_dimensions+col] = 0.0f;
      for(int j=0; j < num_events; j++) {
        if(row==col) {
          components->R[row*num_dimensions+col] += (fcs_data[j*num_dimensions + row])*(fcs_data[j*num_dimensions + row]);
        }
      }
      if(row==col) {
        components->R[row*num_dimensions+col] /= (float) (num_events -1);
        components->R[row*num_dimensions+col] -= ((float)(num_events)*means[row]*means[row]) / (float)(num_events-1);
        components->R[row*num_dimensions+col] /= (float)num_components;
      }
    }
}

void average_variance${'_'+'_'.join(param_val_list)}(float* fcs_data, float* means, int num_dimensions, int num_events, float* avgvar) {
    
    cilk::reducer_opadd<float> total(0.0f);
    // Compute average variance for each dimension
    cilk_for(int i = 0; i < num_dimensions; i++) {
        float variance = 0.0f;
        for(int j=0; j < num_events; j++) {
            variance += fcs_data[j*num_dimensions + i]*fcs_data[j*num_dimensions + i];
        }
        variance /= (float) num_events;
        variance -= means[i]*means[i];
        total += variance;
    }
    
    *avgvar = total.get_value() / (float) num_dimensions;
}

void constants${'_'+'_'.join(param_val_list)}(components_t* components, int M, int D) {
    float log_determinant;
    float* matrix = (float*) malloc(sizeof(float)*D*D);

    //float sum = 0.0;
    for(int m=0; m < M; m++) {
        // Invert covariance matrix
        memcpy(matrix,&(components->R[m*D*D]),sizeof(float)*D*D);
        invert_cpu(matrix,D,&log_determinant);
        memcpy(&(components->Rinv[m*D*D]),matrix,sizeof(float)*D*D);
    
        // Compute constant
        components->constant[m] = -D*0.5f*logf(2*PI) - 0.5f*log_determinant;
        components->CP[m] = components->constant[m]*2.0;
    }
    normalize_pi(components, M);
    free(matrix);
}

void seed_components${'_'+'_'.join(param_val_list)}(float *data_by_event, components_t* components, int num_dimensions, int num_components, int num_events) {
    float* means = (float*) malloc(sizeof(float)*num_dimensions);
    float avgvar;

    // Compute means
    mvtmeans(data_by_event, num_dimensions, num_events, means);

    // Compute the average variance
    seed_covars${'_'+'_'.join(param_val_list)}(components, data_by_event, means, num_dimensions, num_events, &avgvar, num_components);
    average_variance${'_'+'_'.join(param_val_list)}(data_by_event, means, num_dimensions, num_events, &avgvar);    
    float seed;
    if(num_components > 1) {
       seed = (num_events)/(num_components);
    } else {
       seed = 0.0f;
    }

    memcpy(components->means, means, sizeof(float)*num_dimensions);

    for(int c=1; c < num_components; c++) {
        memcpy(&components->means[c*num_dimensions], &data_by_event[((int)(c*seed))*num_dimensions], sizeof(float)*num_dimensions);
          
        for(int i=0; i < num_dimensions*num_dimensions; i++) {
          components->R[c*num_dimensions*num_dimensions+i] = components->R[i];
          components->Rinv[c*num_dimensions*num_dimensions+i] = 0.0f;
        }
    }

    //compute pi, N
    for(int c =0; c<num_components; c++) {
        components->pi[c] = 1.0f/((float)num_components);
        components->N[c] = ((float) num_events) / ((float)num_components);
        components->avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;
    }

    free(means);
}

void compute_average_variance${'_'+'_'.join(param_val_list)}( float* fcs_data, components_t* components, int num_dimensions, int num_components, int num_events)
{
    float* means = (float*) malloc(sizeof(float)*num_dimensions);
    float avgvar;
    
    // Compute the means
    mvtmeans(fcs_data, num_dimensions, num_events, means);
   
    average_variance${'_'+'_'.join(param_val_list)}(fcs_data, means, num_dimensions, num_events, &avgvar);    
    
    for(int c =0; c<num_components; c++) {
        components->avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;
    }
}

void estep1${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N, float* loglikelihoods) {
    // Compute likelihood for every data point in each component
    float* temploglikelihoods = (float*)malloc(M*N*sizeof(float));
    cilk_for(int m=0; m < M; m++) {
        float component_pi = components->pi[m];
        float component_constant = components->constant[m];
        float* means = &(components->means[m*D]);
        float* Rinv = &(components->Rinv[m*D*D]);
        for(int n=0; n < N; n++) {
            float like = 0.0;
%if cvtype == 'diag':
            for(int i=0; i < D; i++) {
                like += (data[i*N+n]-means[i])*(data[i*N+n]-means[i])*Rinv[i*D+i];
            }
%else:
            for(int i=0; i < D; i++) {
                for(int j=0; j < D; j++) {
                    like += (data[i*N+n]-means[i])*(data[j*N+n]-means[j])*Rinv[i*D+j];
                }
            }
%endif
            component_memberships[m*N+n] = (component_pi > 0.0f) ? -0.5*like + component_constant + logf(component_pi) : MINVALUEFORMINUSLOG;
        }
    }
    //estep1 log_add()
    for(int n=0; n < N; n++) {
        float finalloglike = MINVALUEFORMINUSLOG;
        for(int m=0; m < M; m++) {
            finalloglike = log_add(finalloglike, component_memberships[m*N+n]);
        }
        loglikelihoods[n] = finalloglike;
    }
}

float estep2_events${'_'+'_'.join(param_val_list)}(components_t* components, float* component_memberships, int M, int n, int N) {
	// Finding maximum likelihood for this data point
        float temp = 0.0f;
        float thread_likelihood = 0.0f;
	float max_likelihood;
	float denominator_sum = 0.0f;

	max_likelihood = __sec_reduce_max(component_memberships[n:M:N]);
        //max_likelihood = component_memberships[n];
        //for(int m = 1; m < M; m++)
        //    max_likelihood =
        //          fmaxf(max_likelihood,component_memberships[m*N+n]);

	// Computes sum of all likelihoods for this event
	for(int m=0; m < M; m++) {
            temp = expf(component_memberships[m*N+n] - max_likelihood);
            denominator_sum += temp;
	}
	temp = max_likelihood + logf(denominator_sum);
        thread_likelihood += temp;

	// Divide by denominator to get each membership
	for(int m=0; m < M; m++) {
	    component_memberships[m*N+n] = expf(component_memberships[m*N+n] - temp);
	}
        //or component_memberships[n:M:N] = exp(component_memberships[n:M:N] - denominator_sum);

	return thread_likelihood;
}

void estep2${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N, float* likelihood) {
    cilk::reducer_opadd<float> total(0.0f);
    cilk_for(int n=0; n < N; n++) {
        total += estep2_events${'_'+'_'.join(param_val_list)}(components, component_memberships, M, n, N);
    }
    *likelihood = total.get_value();
}

void mstep_mean${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N) {
    cilk_for(int m=0; m < M; m++) {
        for(int d=0; d < D; d++) {
	    components->means[m*D+d] = 0.0;
	    for(int n=0; n < N; n++) {
		components->means[m*D+d] += data[d*N+n]*component_memberships[m*N+n];
	    }
	    components->means[m*D+d] /= components->N[m];
        }
    }
}

void mstep_mean_idx${'_'+'_'.join(param_val_list)}(float* data_by_dimension, int* indices, int num_indices, components_t* components, float* component_memberships, int D, int M, int N) {
    cilk_for(int m=0; m < M; m++) {
        for(int d=0; d < D; d++) {
	    components->means[m*D+d] = 0.0;
	    for(int index = 0; index < num_indices; index++) {
                int n = indices[index];
		components->means[m*D+d] += data_by_dimension[d*N+n]*component_memberships[m*N+n];
	    }
	    components->means[m*D+d] /= components->N[m];
        }
    }
}

void mstep_n${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N) {
    cilk_for(int m=0; m < M; m++) {
        components->N[m] = 0.0;
        for(int n=0; n < N; n++) {
            components->N[m] += component_memberships[m*N+n];
        }
        components->pi[m] =  components->N[m];
    }
}

void mstep_n_idx${'_'+'_'.join(param_val_list)}(float* data, int* indices, int num_indices, components_t* components, float* component_memberships, int D, int M, int N) {
    cilk_for(int m=0; m < M; m++) {
        components->N[m] = 0.0;
        for(int index=0; index < num_indices; index++) {
            int n = indices[index];
            components->N[m] += component_memberships[m*N+n];
        }
        components->pi[m] =  components->N[m];
    }
}

void mstep_covar${'_'+'_'.join(param_val_list)}(float* data, components_t* components,float* component_memberships, int D, int M, int N) {
    cilk_for(int m=0; m < M; m++) {
        float* means = &(components->means[m*D]);
        cilk::reducer_opadd<float> cov_sum(0.0f);
        for(int i=0; i < D; i++) {
            for(int j=0; j <= i; j++) {
%if cvtype == 'diag':
                if(i != j) {
                    components->R[m*D*D+i*D+j] = 0.0f;
                    components->R[m*D*D+j*D+i] = 0.0f;
                    continue;
                }
%endif
                float sum = 0.0;
                for(int n=0; n < N; n++) {
                    sum += (data[i*N+n]-means[i])*(data[j*N+n]-means[j])*component_memberships[m*N+n];
                }

                if(components->N[m] >= 1.0f) {
                    components->R[m*D*D+i*D+j] = sum / components->N[m];
                    components->R[m*D*D+j*D+i] = sum / components->N[m];
                } else {
                    components->R[m*D*D+i*D+j] = 0.0f;
                    components->R[m*D*D+j*D+i] = 0.0f;
                }
                if(i == j) {
                    components->R[m*D*D+j*D+i] += components->avgvar[m];
                }
            }
        }
    }
}

void mstep_covar_idx${'_'+'_'.join(param_val_list)}(float* data_by_dimension, float* data_by_event, int* indices, int num_indices, components_t* components, float* component_memberships, int D, int M, int N) {
    cilk_for(int m=0; m < M; m++) {
        float* means = &(components->means[m*D]);
        cilk::reducer_opadd<float> cov_sum(0.0f);
        for(int i=0; i < D; i++) {
            for(int j=0; j <= i; j++) {
%if cvtype == 'diag':
                if(i != j) {
                    components->R[m*D*D+i*D+j] = 0.0f;
                    components->R[m*D*D+j*D+i] = 0.0f;
                    continue;
                }
%endif
                float sum = 0.0;
                for(int index=0; index < num_indices; index++) {
                    int n = indices[index];
                    sum += (data_by_dimension[i*N+n]-means[i])*(data_by_dimension[j*N+n]-means[j])*component_memberships[m*N+n];
                }

                if(components->N[m] >= 1.0f) {
                    components->R[m*D*D+i*D+j] = sum / components->N[m];
                    components->R[m*D*D+j*D+i] = sum / components->N[m];
                } else {
                    components->R[m*D*D+i*D+j] = 0.0f;
                    components->R[m*D*D+j*D+i] = 0.0f;
                }
                if(i == j) {
                    components->R[m*D*D+j*D+i] += components->avgvar[m];
                }
            }
        }
    }
}
