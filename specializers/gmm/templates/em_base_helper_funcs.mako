#define PI  3.1415926535897931
#define COVARIANCE_DYNAMIC_RANGE 1E6
#define MINVALUEFORMINUSLOG -1000.0

void print_evals(float* component_memberships, float* loglikelihoods, int num_events, int num_components){
  for(int m = 0; m < num_components; m++){
    for(int e = 0; e < num_events; e++)
        printf("%0.8f ", component_memberships[m*num_events+e]);
    printf("\n");
  }
  for(int e = 0; e < num_events; e++)
    printf("%0.8f ", loglikelihoods[e]);
  printf("\n");
}

void print_components(components_t * components, int num_components, int num_dimensions){
  printf("===============\n");
  for(int m = 0; m < num_components; m++){
       printf("%0.8f ", components->N[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
       printf("%0.8f ", components->pi[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
       printf("%0.8f ", components->CP[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
       printf("%0.8f ", components->constant[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
       printf("%0.8f ", components->avgvar[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
    for(int d = 0; d < num_dimensions; d++)
        printf("%0.8f ", components->means[m*num_dimensions+d]);
    printf("\n");
  }
    for(int m = 0; m < num_components; m++){
        for(int d = 0; d < num_dimensions; d++)
            for(int d2 = 0; d2 < num_dimensions; d2++)
                printf("%0.8f ",
                components->R[m*num_dimensions*num_dimensions+d*num_dimensions+d2]);
        printf("\n");
    }

    for(int m = 0; m < num_components; m++){
        for(int d = 0; d < num_dimensions; d++)
            for(int d2 = 0; d2 < num_dimensions; d2++)
                printf("%0.8f ",
                components->Rinv[m*num_dimensions*num_dimensions+d*num_dimensions+d2]);
        printf("\n");
    }
  printf("===============\n");
}

typedef struct return_component_container
{
  boost::python::object component;
  float distance;
} ret_c_con_t;

ret_c_con_t ret;

void mvtmeans(float* data_by_event, int num_dimensions, int num_events, float* means) {
    for(int d=0; d < num_dimensions; d++) {
        means[d] = 0.0;
        for(int n=0; n < num_events; n++) {
            means[d] += data_by_event[n*num_dimensions+d];
        }
        means[d] /= (float) num_events;
    }
}

float log_add(float log_a, float log_b) {
  if(log_a < log_b) {
      float tmp = log_a;
      log_a = log_b;
      log_b = tmp;
    }
  //setting MIN...LOG so small, I don't even need to look
  return (((log_b - log_a) <= MINVALUEFORMINUSLOG) ? log_a : 
                log_a + (float)(logf(1.0 + (double)(expf((double)(log_b - log_a))))));
}

void normalize_pi(components_t* components, int num_components) {
    float total = 0;
    for(int i=0; i<num_components; i++) {
        total += components->pi[i];
    }
    
    for(int m=0; m < num_components; m++){
        components->pi[m] /= total; 
    }
}
//=== Data structure pointers ===

//CPU copies of events
float *fcs_data_by_event;
float *fcs_data_by_dimension;

// index list for train_on_subset
int* index_list;


//CPU copies of components
components_t components;
components_t saved_components;
components_t** scratch_component_arr; // for computing distances and merging
static int num_scratch_components = 0;

//CPU copies of eval data
float *component_memberships;
float *loglikelihoods;

//=== AHC function prototypes ===
void copy_component(components_t *dest, int c_dest, components_t *src, int c_src, int num_dimensions);
void add_components(components_t *components, int c1, int c2, components_t *temp_component, int num_dimensions);
float component_distance(components_t *components, int c1, int c2, components_t *temp_component, int num_dimensions);

//=== Helper function prototypes ===
void writeCluster(FILE* f, components_t components, int c,  int num_dimensions);
void printCluster(components_t components, int c, int num_dimensions);
void invert_cpu(float* data, int actualsize, float* log_determinant);
int  invert_matrix(float* a, int n, float* determinant);

//============ LUTLOG ==============

float *LOOKUP_TABLE;
int N_LOOKUP_SIZE = 12;

void do_table(int n,float *lookup_table)
{
  float numlog;
  int *const exp_ptr = ((int*)&numlog);
  int x = *exp_ptr;
  x = 0x00000000;
  x += 127 << 23;
  *exp_ptr = x;
  for(int i=0;i<pow((double) 2,(double) n);i++)
    {
      lookup_table[i]=log2(numlog);
      x+=1 << (23-n);
      *exp_ptr = x;
    }
}

void create_lut_log_table() {

  unsigned int tablesize = (unsigned int)pow(2.0, 12);
  LOOKUP_TABLE = (float*) malloc(tablesize*sizeof(float));
  do_table(N_LOOKUP_SIZE,LOOKUP_TABLE);

}

//================ END LUTLOG ============


//=== Memory Alloc/Free Functions ===

components_t* alloc_temp_component_on_CPU(int num_dimensions) {

  components_t* scratch_component = (components_t*)malloc(sizeof(components_t));

  scratch_component->N = (float*) malloc(sizeof(float));
  scratch_component->pi = (float*) malloc(sizeof(float));
  scratch_component->CP = (float*) malloc(sizeof(float));
  scratch_component->constant = (float*) malloc(sizeof(float));
  scratch_component->avgvar = (float*) malloc(sizeof(float));
  scratch_component->means = (float*) malloc(sizeof(float)*num_dimensions);
  scratch_component->R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
  scratch_component->Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);

  return scratch_component;
}

void dealloc_temp_components_on_CPU() {
  printf("dealloc tempcomponents on CPU\n");
  for(int i = 0; i<num_scratch_components; i++) {
    free(scratch_component_arr[i]->N);
    free(scratch_component_arr[i]->pi);
    free(scratch_component_arr[i]->CP);
    free(scratch_component_arr[i]->constant);
    free(scratch_component_arr[i]->avgvar);
    free(scratch_component_arr[i]->means);
    free(scratch_component_arr[i]->R);
    free(scratch_component_arr[i]->Rinv);
  }
  num_scratch_components = 0;

  return;
}


// ================== Event data allocation on CPU  ================= :
void alloc_events_on_CPU(PyObject *input_data) {

  fcs_data_by_event = ((float*)PyArray_DATA(input_data));
  int num_events = PyArray_DIM(input_data,0);
  int num_dimensions = PyArray_DIM(input_data,1);
  // Transpose the event data (allows coalesced access pattern in E-step kernel)
  // This has consecutive values being from the same dimension of the data
  // (num_dimensions by num_events matrix)
  fcs_data_by_dimension  = (float*) malloc(sizeof(float)*num_events*num_dimensions);

  for(int e=0; e<num_events; e++) {
    for(int d=0; d<num_dimensions; d++) {
      fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
    }
  }
}

void alloc_index_list_on_CPU(PyObject *input_index_list) {
  index_list =  ((int*)PyArray_DATA(input_index_list));
}

void alloc_events_from_index_on_CPU(PyObject *input_data, PyObject *indices, int num_indices, int num_dimensions) {

  fcs_data_by_event = (float*)malloc(num_indices*num_dimensions*sizeof(int));
  for(int i = 0; i<num_indices; i++) {
    for(int d = 0; d<num_dimensions; d++) {
      fcs_data_by_event[i*num_dimensions+d] = ((float*)PyArray_DATA(input_data))[((int*)PyArray_DATA(indices))[i]*num_dimensions+d];
    }
  }

  fcs_data_by_dimension = (float*)malloc(num_indices*num_dimensions*sizeof(int));
  for(int e=0; e<num_indices; e++) {
    for(int d = 0; d<num_dimensions; d++) {
      fcs_data_by_dimension[d*num_indices+e] = fcs_data_by_event[e*num_dimensions+d];
      //printf("data: %f\n", fcs_data_by_dimension[d*num_indices+e]);
    }
  }
}

// ================== Cluster data allocation on CPU  ================= :

void alloc_components_on_CPU(int M, int D, PyObject *weights, PyObject *means, PyObject *covars, PyObject *comp_probs) {
  //printf("IN C = alloc components on cpu\n");
  components.pi = ((float*)PyArray_DATA(weights));
  components.means = ((float*)PyArray_DATA(means));
  components.R = ((float*)PyArray_DATA(covars));
  components.CP = ((float*)PyArray_DATA(comp_probs));
  
  components.N = (float*) malloc(sizeof(float)*M);
  components.constant = (float*) malloc(sizeof(float)*M);
  components.avgvar = (float*) malloc(sizeof(float)*M);
  components.Rinv = (float*) malloc(sizeof(float)*M*D*D);
}  

//Hacky way to make sure the CPU pointers are aimed at the right component data
void relink_components_on_CPU(PyObject *weights, PyObject *means, PyObject *covars) {
  components.pi = ((float*)PyArray_DATA(weights));
  components.means = ((float*)PyArray_DATA(means));
  components.R = ((float*)PyArray_DATA(covars));
}

// ================= Eval data alloc on CPU =============== 

void alloc_evals_on_CPU(PyObject *component_mem_np_arr, PyObject *loglikelihoods_np_arr){
  component_memberships = ((float*)PyArray_DATA(component_mem_np_arr));
  loglikelihoods = ((float*)PyArray_DATA(loglikelihoods_np_arr));
}

// ================== Event data dellocation on CPU  ================= :
void dealloc_events_on_CPU() {
  //free(fcs_data_by_event);
  free(fcs_data_by_dimension);
 return;
}

// ================== Index list dellocation on CPU  ================= :
void dealloc_index_list_on_CPU() {
  free(index_list);
  return;
}

// ==================== Cluster data deallocation on CPU =================  
void dealloc_components_on_CPU() {

  //free(components.pi);
  //free(components.means);
  //free(components.R);
  //free(components.CP);
  free(components.N);
  free(components.constant);
  free(components.avgvar);
  free(components.Rinv);
  return;
}

// ==================== Eval data deallocation on CPU =================  
void dealloc_evals_on_CPU() {
  //free(component_memberships);
  //free(loglikelihoods);
  return;
}


// ==== Accessor functions for pi, means, covars ====

PyObject *get_temp_component_pi(components_t* c){
  npy_intp dims[1] = {1};
  return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, c->pi);
}

PyObject *get_temp_component_means(components_t* c, int D){
  npy_intp dims[1] = {D};
  return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, c->means);
}

PyObject *get_temp_component_covars(components_t* c, int D){
  npy_intp dims[2] = {D, D};
  return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, c->R);
}

//------------------------- AHC FUNCTIONS ----------------------------

//============ KL DISTANCE FUNCTIONS =============
inline float lut_log (float val, float *lookup_table, int n)
{
  int *const     exp_ptr = ((int*)&val);
  int            x = *exp_ptr;
  const int      log_2 = ((x >> 23) & 255) - 127;
  x &= 0x7FFFFF;
  x = x >> (23-n);
  val=lookup_table[x];
  // printf("log2:%f\n", log_2);
  return ((val + log_2)* 0.69314718);

}

// sequentuially add logarithms
float Log_Add(float log_a, float log_b)
{
  float result;
  if(log_a < log_b)
    {
      float tmp = log_a;
      log_a = log_b;
      log_b = tmp;
    }
  //setting MIN...LOG so small, I don't even need to look
  if((log_b - log_a) <= MINVALUEFORMINUSLOG)
    {
      return log_a;
    }
  else
    {
      result = log_a + (float)(lut_log(1.0 + (double)(exp((double)(log_b - log_a))),LOOKUP_TABLE,N_LOOKUP_SIZE));
    }
  return result;
}

double Log_Likelihood(int DIM, int m, float *feature, float *means, float *covars, float CP)
{
  //float log_lkld;
  //float in_the_exp = 0.0, den = 0.0;
  double x,y=0,z;
  for(int i=0; i<DIM; i++)
    {
      x = feature[i]-means[DIM*m + i];
      z = covars[m*DIM*DIM + i*DIM+i];
      y += x*x/z;//+lut_log(2*3.141592654*z,LOOKUP_TABLE,N_LOOKUP_SIZE); LINE MODIFIED
      // printf("y = %f, feature[%d]  = %f, mean[%d] = %f \n", y, i, feature[i], i, means[i*m+i], m*DIM*DIM+i*DIM+i, cov\
      ars[m*DIM*DIM + i*DIM+i]);
}
//printf("y  = %f, CP  = %f\n", y, CP);
return((double)-0.5*(y+CP)); //LINE MODIFIED
}


float Log_Likelihood_KL(float *feature, int DIM, int gmm_M, float *gmm_weights, float *gmm_means, float *gmm_covars, float *gmm_CP)
{

  //float res = 0.0;
  float log_lkld= MINVALUEFORMINUSLOG ,aux;
  for(int i=0;i<gmm_M;i++)
    {
      // if(gmm_weights[i])
      // {
      aux = lut_log(gmm_weights[i],LOOKUP_TABLE,N_LOOKUP_SIZE) + Log_Likelihood(DIM, i, feature, gmm_means, gmm_covars, gmm_CP[i]);


      if(isnan(aux) || !finite(aux))
        {
          aux = MINVALUEFORMINUSLOG;
        }
      log_lkld = Log_Add(log_lkld, aux);
      //}
    }//for
  return log_lkld;
}


float compute_KL_distance(int DIM, int gmm1_M, int gmm2_M, PyObject *gmm1_weights_in, PyObject *gmm1_means_in, PyObject *gmm1_covars_in, PyObject *gmm1_CP_in, PyObject *gmm2_weights_in, PyObject *gmm2_means_in, PyObject *gmm2_covars_in, PyObject *gmm2_CP_in) {

  float aux;
  float log_g1,log_f1,log_g2,log_f2,f_log_g=0,f_log_f=0,g_log_f=0,g_log_g=0;
  float *point_a = new float[DIM];
  float *point_b = new float[DIM];

  float *gmm1_weights = ((float*)PyArray_DATA(gmm1_weights_in));
  float *gmm1_means = ((float*)PyArray_DATA(gmm1_means_in));
  float *gmm1_covars = ((float*)PyArray_DATA(gmm1_covars_in));
  float *gmm1_CP = ((float*)PyArray_DATA(gmm1_CP_in));
  float *gmm2_weights = ((float*)PyArray_DATA(gmm2_weights_in));
  float *gmm2_means = ((float*)PyArray_DATA(gmm2_means_in));
  float *gmm2_covars = ((float*)PyArray_DATA(gmm2_covars_in));
  float *gmm2_CP = ((float*)PyArray_DATA(gmm2_CP_in));

  for(int i=0;i<gmm1_M;i++)
    {
      log_g1=0;
      log_f1=0;
      for(int k=0;k<DIM;k++)
        {
          //Compute the two points
          for(int j=0;j<DIM;j++)
            {
              if(j==k){
                aux = sqrt(19.0)*sqrt(gmm1_covars[i*DIM*DIM + k*DIM+k]);
                point_a[j] = gmm1_means[i*DIM+j] + aux;
                point_b[j] = gmm1_means[i*DIM+j] - aux;
              }
              else{
                point_a[j] = gmm1_means[i*DIM+j];
                point_b[j] = gmm1_means[i*DIM+j];
              }
            }
          log_g1+=Log_Likelihood_KL(point_a, DIM, gmm2_M, gmm2_weights, gmm2_means, gmm2_covars, gmm2_CP)+Log_Likelihood_KL(point_b, DIM, gmm2_M, gmm2_weights, gmm2_means, gmm2_covars, gmm2_CP);
          log_f1+=Log_Likelihood_KL(point_a, DIM, gmm1_M, gmm1_weights, gmm1_means, gmm1_covars, gmm1_CP)+Log_Likelihood_KL(point_b, DIM, gmm1_M, gmm1_weights, gmm1_means, gmm1_covars, gmm1_CP);
        }

      f_log_g+=gmm1_weights[i]*log_g1;
      f_log_f+=gmm1_weights[i]*log_f1;

    }
  for(int i=0;i<gmm2_M;i++)
    {
      log_g2=0;
      log_f2=0;
      for(int k=0;k<DIM;k++)
        {
          for(int j=0;j<DIM;j++)
            {
              if(j==k){
                aux = sqrt(19.0)*sqrt(gmm2_covars[i*DIM*DIM + k*DIM+k]);
                point_a[j] = gmm2_means[i*DIM+j] + aux;
                point_b[j] = gmm2_means[i*DIM+j] - aux;
              }
              else{
                point_a[j] = gmm2_means[i*DIM+j];
                point_b[j] = gmm2_means[i*DIM+j];
              }
            }

          log_g2+=Log_Likelihood_KL(point_a, DIM, gmm2_M, gmm2_weights, gmm2_means, gmm2_covars, gmm2_CP)+Log_Likelihood_KL(point_b, DIM, gmm2_M, gmm2_weights, gmm2_means, gmm2_covars, gmm2_CP);
          log_f2+=Log_Likelihood_KL(point_a, DIM, gmm1_M, gmm1_weights, gmm1_means, gmm1_covars, gmm1_CP)+Log_Likelihood_KL(point_b, DIM, gmm1_M, gmm1_weights, gmm1_means, gmm1_covars, gmm1_CP);
        }
      g_log_g+=gmm2_weights[i]*log_g2;
      g_log_f+=gmm2_weights[i]*log_f2;

    }
  delete [] point_a;
  delete [] point_b;
  return 1.0/(2.0*DIM)*(f_log_f + g_log_g - f_log_g - g_log_f);
}


int compute_distance_rissanen(int c1, int c2, int num_dimensions) {
  // compute distance function between the 2 components

  components_t *new_component = alloc_temp_component_on_CPU(num_dimensions);

  float distance = component_distance(&components,c1,c2,new_component,num_dimensions);
  //printf("distance %d-%d: %f\n", c1, c2, distance);

  scratch_component_arr[num_scratch_components] = new_component;
  num_scratch_components++;
  
  ret.component = boost::python::object(boost::python::ptr(new_component));
  ret.distance = distance;

  return 0;

}

void merge_components(int min_c1, int min_c2, components_t *min_component, int num_components, int num_dimensions) {

  // Copy new combined component into the main group of components, compact them
  copy_component(&components,min_c1, min_component,0,num_dimensions);

  for(int i=min_c2; i < num_components-1; i++) {
  
    copy_component(&components,i,&components,i+1,num_dimensions);
  }
}


float component_distance(components_t *components, int c1, int c2, components_t *temp_component, int num_dimensions) {
  // Add the components together, this updates pi,means,R,N and stores in temp_component

  add_components(components,c1,c2,temp_component,num_dimensions);
  //printf("%f, %f, %f, %f, %f, %f\n", components->N[c1], components->constant[c1], components->N[c2], components->constant[c2], temp_component->N[0], temp_component->constant[0]);
  return components->N[c1]*components->constant[c1] + components->N[c2]*components->constant[c2] - temp_component->N[0]*temp_component->constant[0];
  
}

void add_components(components_t *components, int c1, int c2, components_t *temp_component, int num_dimensions) {
  float wt1,wt2;
 
  wt1 = (components->N[c1]) / (components->N[c1] + components->N[c2]);
  wt2 = 1.0f - wt1;
    
  // Compute new weighted means
  for(int i=0; i<num_dimensions;i++) {
    temp_component->means[i] = wt1*components->means[c1*num_dimensions+i] + wt2*components->means[c2*num_dimensions+i];
  }
    
  // Compute new weighted covariance
  for(int i=0; i<num_dimensions; i++) {
    for(int j=i; j<num_dimensions; j++) {
      // Compute R contribution from component1
      temp_component->R[i*num_dimensions+j] = ((temp_component->means[i]-components->means[c1*num_dimensions+i])
                                             *(temp_component->means[j]-components->means[c1*num_dimensions+j])
                                             +components->R[c1*num_dimensions*num_dimensions+i*num_dimensions+j])*wt1;
      // Add R contribution from component2
      temp_component->R[i*num_dimensions+j] += ((temp_component->means[i]-components->means[c2*num_dimensions+i])
                                              *(temp_component->means[j]-components->means[c2*num_dimensions+j])
                                              +components->R[c2*num_dimensions*num_dimensions+i*num_dimensions+j])*wt2;
      // Because its symmetric...
      temp_component->R[j*num_dimensions+i] = temp_component->R[i*num_dimensions+j];
    }
  }
    
  // Compute pi
  temp_component->pi[0] = components->pi[c1] + components->pi[c2];
    
  // compute N
  temp_component->N[0] = components->N[c1] + components->N[c2];

  float log_determinant;
  // Copy R to Rinv matrix
  memcpy(temp_component->Rinv,temp_component->R,sizeof(float)*num_dimensions*num_dimensions);
  // Invert the matrix
  invert_cpu(temp_component->Rinv,num_dimensions,&log_determinant);
  // Compute the constant
  temp_component->constant[0] = (-num_dimensions)*0.5*logf(2*PI)-0.5*log_determinant;
    
  // avgvar same for all components
  temp_component->avgvar[0] = components->avgvar[0];
}

void copy_component(components_t *dest, int c_dest, components_t *src, int c_src, int num_dimensions) {
  dest->N[c_dest] = src->N[c_src];
  dest->pi[c_dest] = src->pi[c_src];
  dest->constant[c_dest] = src->constant[c_src];
  dest->avgvar[c_dest] = src->avgvar[c_src];
  memcpy(&(dest->means[c_dest*num_dimensions]),&(src->means[c_src*num_dimensions]),sizeof(float)*num_dimensions);
  memcpy(&(dest->R[c_dest*num_dimensions*num_dimensions]),&(src->R[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  memcpy(&(dest->Rinv[c_dest*num_dimensions*num_dimensions]),&(src->Rinv[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  // do we need to copy memberships?
}
//---------------- END AHC FUNCTIONS ----------------


void writeCluster(FILE* f, components_t components, int c, int num_dimensions) {
  fprintf(f,"Probability: %f\n", components.pi[c]);
  fprintf(f,"N: %f\n",components.N[c]);
  fprintf(f,"Means: ");
  for(int i=0; i<num_dimensions; i++){
    fprintf(f,"%.3f ",components.means[c*num_dimensions+i]);
  }
  fprintf(f,"\n");

  fprintf(f,"\nR Matrix:\n");
  for(int i=0; i<num_dimensions; i++) {
    for(int j=0; j<num_dimensions; j++) {
      fprintf(f,"%.3f ", components.R[c*num_dimensions*num_dimensions+i*num_dimensions+j]);
    }
    fprintf(f,"\n");
  }
  fflush(f);   
  /*
    fprintf(f,"\nR-inverse Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
    for(int j=0; j<num_dimensions; j++) {
    fprintf(f,"%.3f ", c->Rinv[i*num_dimensions+j]);
    }
    fprintf(f,"\n");
    } 
  */
}

void printCluster(components_t components, int c, int num_dimensions) {
  writeCluster(stdout,components,c,num_dimensions);
}

static int 
ludcmp(float *a,int n,int *indx,float *d);

static void 
lubksb(float *a,int n,int *indx,float *b);

/*
 * Inverts a square matrix (stored as a 1D float array)
 * 
 * actualsize - the dimension of the matrix
 *
 * written by Mike Dinolfo 12/98
 * version 1.0
 */
void invert_cpu(float* data, int actualsize, float* log_determinant)  {
  int maxsize = actualsize;
  int n = actualsize;
  *log_determinant = 0.0;

  if (actualsize == 1) { // special case, dimensionality == 1
    *log_determinant = logf(data[0]);
    data[0] = 1.0 / data[0];
  } else if(actualsize >= 2) { // dimensionality >= 2
    for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
    for (int i=1; i < actualsize; i++)  { 
      for (int j=i; j < actualsize; j++)  { // do a column of L
        float sum = 0.0;
        for (int k = 0; k < i; k++)  
          sum += data[j*maxsize+k] * data[k*maxsize+i];
        data[j*maxsize+i] -= sum;
      }
      if (i == actualsize-1) continue;
      for (int j=i+1; j < actualsize; j++)  {  // do a row of U
        float sum = 0.0;
        for (int k = 0; k < i; k++)
          sum += data[i*maxsize+k]*data[k*maxsize+j];
        data[i*maxsize+j] = 
          (data[i*maxsize+j]-sum) / data[i*maxsize+i];
      }
    }

    for(int i=0; i<actualsize; i++) {
      *log_determinant += logf(fabs(data[i*n+i]));
      //printf("log_determinant: %e\n",*log_determinant); 
    }
    //printf("\n\n");
    for ( int i = 0; i < actualsize; i++ )  // invert L
      for ( int j = i; j < actualsize; j++ )  {
        float x = 1.0;
        if ( i != j ) {
          x = 0.0;
          for ( int k = i; k < j; k++ ) 
            x -= data[j*maxsize+k]*data[k*maxsize+i];
        }
        data[j*maxsize+i] = x / data[j*maxsize+j];
      }
    for ( int i = 0; i < actualsize; i++ )   // invert U
      for ( int j = i; j < actualsize; j++ )  {
        if ( i == j ) continue;
        float sum = 0.0;
        for ( int k = i; k < j; k++ )
          sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
        data[i*maxsize+j] = -sum;
      }
    for ( int i = 0; i < actualsize; i++ )   // final inversion
      for ( int j = 0; j < actualsize; j++ )  {
        float sum = 0.0;
        for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
          sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
        data[j*maxsize+i] = sum;
      }
  } else {
    printf("Error: Invalid dimensionality for invert(...)\n");
  }
}


/*
 * Another matrix inversion function
 * This was modified from the 'component' application by Charles A. Bouman
 */
int invert_matrix(float* a, int n, float* determinant) {
  int  i,j,f,g;
   
  float* y = (float*) malloc(sizeof(float)*n*n);
  float* col = (float*) malloc(sizeof(float)*n);
  int* indx = (int*) malloc(sizeof(int)*n);
  /*
    printf("\n\nR matrix before LU decomposition:\n");
    for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
    printf("%.2f ",a[i*n+j]);
    }
    printf("\n");
    }*/

  *determinant = 0.0;
  if(ludcmp(a,n,indx,determinant)) {
    printf("Determinant mantissa after LU decomposition: %f\n",*determinant);
    printf("\n\nR matrix after LU decomposition:\n");
    for(i=0; i<n; i++) {
      for(j=0; j<n; j++) {
        printf("%.2f ",a[i*n+j]);
      }
      printf("\n");
    }
       
    for(j=0; j<n; j++) {
      *determinant *= a[j*n+j];
    }
     
    printf("determinant: %E\n",*determinant);
     
    for(j=0; j<n; j++) {
      for(i=0; i<n; i++) col[i]=0.0;
      col[j]=1.0;
      lubksb(a,n,indx,col);
      for(i=0; i<n; i++) y[i*n+j]=col[i];
    }

    for(i=0; i<n; i++)
      for(j=0; j<n; j++) a[i*n+j]=y[i*n+j];
     
    printf("\n\nMatrix at end of clust_invert function:\n");
    for(f=0; f<n; f++) {
      for(g=0; g<n; g++) {
        printf("%.2f ",a[f*n+g]);
      }
      printf("\n");
    }
    free(y);
    free(col);
    free(indx);
    return(1);
  }
  else {
    *determinant = 0.0;
    free(y);
    free(col);
    free(indx);
    return(0);
  }
}

#define TINY 1.0e-20

static int
ludcmp(float *a,int n,int *indx,float *d)
{
  int i,imax=0,j,k;
  float big,dum,sum,temp;
  float *vv;

  vv= (float*) malloc(sizeof(float)*n);
   
  *d=1.0;
   
  for (i=0;i<n;i++)
    {
      big=0.0;
      for (j=0;j<n;j++)
        if ((temp=fabsf(a[i*n+j])) > big)
          big=temp;
      if (big == 0.0)
        return 0; /* Singular matrix  */
      vv[i]=1.0/big;
    }
       
   
  for (j=0;j<n;j++)
    {  
      for (i=0;i<j;i++)
        {
          sum=a[i*n+j];
          for (k=0;k<i;k++)
            sum -= a[i*n+k]*a[k*n+j];
          a[i*n+j]=sum;
        }
       
      /*
        int f,g;
        printf("\n\nMatrix After Step 1:\n");
        for(f=0; f<n; f++) {
        for(g=0; g<n; g++) {
        printf("%.2f ",a[f*n+g]);
        }
        printf("\n");
        }*/
       
      big=0.0;
      dum=0.0;
      for (i=j;i<n;i++)
        {
          sum=a[i*n+j];
          for (k=0;k<j;k++)
            sum -= a[i*n+k]*a[k*n+j];
          a[i*n+j]=sum;
          dum=vv[i]*fabsf(sum);
          //printf("sum: %f, dum: %f, big: %f\n",sum,dum,big);
          //printf("dum-big: %E\n",fabs(dum-big));
          if ( (dum-big) >= 0.0 || fabs(dum-big) < 1e-3)
            {
              big=dum;
              imax=i;
              //printf("imax: %d\n",imax);
            }
        }
       
      if (j != imax)
        {
          for (k=0;k<n;k++)
            {
              dum=a[imax*n+k];
              a[imax*n+k]=a[j*n+k];
              a[j*n+k]=dum;
            }
          *d = -(*d);
          vv[imax]=vv[j];
        }
      indx[j]=imax;
       
      /*
        printf("\n\nMatrix after %dth iteration of LU decomposition:\n",j);
        for(f=0; f<n; f++) {
        for(g=0; g<n; g++) {
        printf("%.2f ",a[f][g]);
        }
        printf("\n");
        }
        printf("imax: %d\n",imax);
      */


      /* Change made 3/27/98 for robustness */
      if ( (a[j*n+j]>=0)&&(a[j*n+j]<TINY) ) a[j*n+j]= TINY;
      if ( (a[j*n+j]<0)&&(a[j*n+j]>-TINY) ) a[j*n+j]= -TINY;

      if (j != n-1)
        {
          dum=1.0/(a[j*n+j]);
          for (i=j+1;i<n;i++)
            a[i*n+j] *= dum;
        }
    }
  free(vv);
  return(1);
}

#undef TINY

static void
lubksb(float *a,int n,int *indx,float *b)
{
  int i,ii,ip,j;
  float sum;

  ii = -1;
  for (i=0;i<n;i++)
    {
      ip=indx[i];
      sum=b[ip];
      b[ip]=b[i];
      if (ii >= 0)
        for (j=ii;j<i;j++)
          sum -= a[i*n+j]*b[j];
      else if (sum)
        ii=i;
      b[i]=sum;
    }
  for (i=n-1;i>=0;i--)
    {
      sum=b[i];
      for (j=i+1;j<n;j++)
        sum -= a[i*n+j]*b[j];
      b[i]=sum/a[i*n+i];
    }
}

