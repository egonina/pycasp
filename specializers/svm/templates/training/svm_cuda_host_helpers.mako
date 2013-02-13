// ============ TRAIN Data Structures =========
// GPU data structure pointers
float* devData;
float* devTransposedData;
size_t devTransposedDataPitch;
float* devLabels;
float* devKernelDiag;
float* devAlphaT;
float* devF;
void* devResultT;

// GPU Cache
float* devCache;
size_t cachePitch;
int devCachePitchInFloats;

//helper data structures
float* devLocalFsRL;
float* devLocalFsRH;
int* devLocalIndicesRL;
int* devLocalIndicesRH;
float* devLocalObjsMaxObj;
int* devLocalIndicesMaxObj;
size_t rowPitch;

// ============ CLASSIFY Data Structures =========
float* devSV;
size_t devSVPitch;
int devSVPitchInFloats;
float* devResultC;

float* devAlphaC;

// For now, assume data, labels and alphas are passed from Python
// all other data structures are internal...
// ========================== GPU Data Allocation Functions ==========================
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

// ========================== TRAIN ===============================

void align_host_data(int nPoints, int nDimension) {
    hostPitchInFloats = nPoints;
    if (devDataPitch == nPoints * sizeof(float)) {
        //printf("Data is already aligned\n");
        hostData = data;
        hostDataAlloced = false;
    } else {
        hostPitchInFloats = devDataPitch/sizeof(float);	
        //printf("Realigning data to pitch: %d\n", devDataPitch);
        hostData = (float*)malloc(devDataPitch * nDimension);
        hostDataAlloced = true;
        for(int i=nDimension-1; i>=0; i--) {
            for(int j=nPoints-1; j>=0; j--) {
                hostData[i * hostPitchInFloats + j]=data[i * nPoints + j];
              }
          }
      }
}

void alloc_transposed_point_data_on_CPU(int nPoints, int nDimension) {
    // Transpose training data on the CPU
    transposedData = (float*)malloc(sizeof(float) * nPoints * nDimension);
    for(int i = 0; i < nPoints; i++) {
        for (int j = 0; j < nDimension; j++) {
            transposedData[i * nDimension + j] = hostData[j * hostPitchInFloats + i];
        }
    }
}

void alloc_transposed_point_data_on_GPU(int nPoints, int nDimension) {
    // Allocate transposed training data on the GPU
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&devTransposedData,
                   &devTransposedDataPitch,
                   nDimension * sizeof(float),
                   nPoints));
    CUT_CHECK_ERROR("Alloc transposed point data on GPU failed: ");
}

void copy_transposed_point_data_CPU_to_GPU(int nPoints, int nDimension) {
    // Copy transposed training data to the GPU
    cudaMemcpy2D(devTransposedData,
                 devTransposedDataPitch,
                 transposedData,
                 nDimension * sizeof(float),
                 nDimension * sizeof(float),
      			 nPoints,
                 cudaMemcpyHostToDevice);
    CUT_CHECK_ERROR("Copy transposed point data from CPU to GPU failed: ");
}

void alloc_point_data_on_GPU(int nPoints, int nDimension) {

    // Allocate training data on the GPU
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&devData, &devDataPitch,
                   nPoints * sizeof(float), nDimension));
    CUT_CHECK_ERROR("Alloc point data on GPU failed: ");
    align_host_data(nPoints, nDimension);
    alloc_transposed_point_data_on_CPU(nPoints, nDimension);
    alloc_transposed_point_data_on_GPU(nPoints, nDimension);
    copy_transposed_point_data_CPU_to_GPU(nPoints, nDimension);
}

void alloc_point_data_on_GPU_from_ptr(size_t ptr, int nPoints, int nDimension) {

    devData = (float*)ptr;
    devDataPitch = (size_t)(nPoints * sizeof(float));
    align_host_data(nPoints, nDimension);
    alloc_transposed_point_data_on_CPU(nPoints, nDimension);
    alloc_transposed_point_data_on_GPU(nPoints, nDimension);
    copy_transposed_point_data_CPU_to_GPU(nPoints, nDimension);
}

void copy_point_data_CPU_to_GPU(int nDimension) {
    // Copy the training data to the GPU
    CUDA_SAFE_CALL(cudaMemcpy(devData, hostData,
                             devDataPitch * nDimension,
                             cudaMemcpyHostToDevice));
    CUT_CHECK_ERROR("Copy point data from CPU to GPU failed: ");
}

void alloc_labels_on_GPU(int nPoints) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLabels, nPoints * sizeof(float)));
    CUT_CHECK_ERROR("Alloc labels on GPU failed: ");
}

void copy_labels_CPU_to_GPU(int nPoints) {
    CUDA_SAFE_CALL(cudaMemcpy(devLabels, labels, nPoints * sizeof(float),
                   cudaMemcpyHostToDevice));
    CUT_CHECK_ERROR("Copy labels from CPU to GPU failed: ");
}

void alloc_train_alphas_on_GPU(int nPoints) {
    // Allocate support vectors on the GPU
    CUDA_SAFE_CALL(cudaMalloc((void**)&devAlphaT, nPoints * sizeof(float)));
    CUT_CHECK_ERROR("Alloc train alphas on GPU failed: ");
}

void alloc_train_result_on_GPU() {
    CUDA_SAFE_CALL(cudaMalloc(&devResultT, 8 * sizeof(float)));
    CUT_CHECK_ERROR("Alloc train result on GPU failed: ");
}

// ========================== CLASSIFY ===============================

void alloc_support_vectors_on_GPU(int nSV, int nDimension) {
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&devSV, &devSVPitch,
                   nSV * sizeof(float), nDimension));
	devSVPitchInFloats = ((int)devSVPitch) / sizeof(float);
    CUT_CHECK_ERROR("Alloc SVs on GPU failed: ");
}

void copy_support_vectors_CPU_to_GPU(int nSV, int nDimension) {
	CUDA_SAFE_CALL(cudaMemcpy2D(devSV, devSVPitch,
                   support_vectors, nSV*sizeof(float),
                   nSV * sizeof(float), nDimension, cudaMemcpyHostToDevice));
    CUT_CHECK_ERROR("Copy SVs to GPU failed: ");
}

void alloc_classify_alphas_on_GPU(int nSV) {
	CUDA_SAFE_CALL(cudaMalloc((void**)&devAlphaC, nSV * sizeof(float)));
    CUT_CHECK_ERROR("Alloc classify alphas on GPU failed: ");
}

void copy_classify_alphas_CPU_to_GPU(int nSV) {
	CUDA_SAFE_CALL(cudaMemcpy(devAlphaC, alphaC, nSV * sizeof(float),
                   cudaMemcpyHostToDevice));
    CUT_CHECK_ERROR("Copy classify alphas to GPU failed: ");
}

void alloc_classify_result_on_GPU(int nPoints) {
	CUDA_SAFE_CALL(cudaMalloc((void**)&devResultC, nPoints * sizeof(float)));
    CUT_CHECK_ERROR("Alloc classify result on GPU failed: ");
}

void dealloc_transposed_point_data_on_CPU() {
   free(transposedData);
}

void dealloc_transposed_point_data_on_GPU(){
    cudaFree(devTransposedData);
    CUT_CHECK_ERROR("Dealloc transposed point data on GPU failed: ");
}

void dealloc_host_data_on_CPU() {
    if (hostDataAlloced) {
      free(hostData);
    }
}

void dealloc_point_data_on_GPU(){
    cudaFree(devData);
    dealloc_transposed_point_data_on_CPU();
    dealloc_transposed_point_data_on_GPU();
    dealloc_host_data_on_CPU();
    CUT_CHECK_ERROR("Dealloc point data on GPU failed: ");
}

void dealloc_labels_on_GPU(){
    cudaFree(devLabels);
    CUT_CHECK_ERROR("Dealloc labels on GPU failed: ");
}

void dealloc_train_alphas_on_GPU(){
    cudaFree(devAlphaT);
    CUT_CHECK_ERROR("Dealloc train alphas on GPU failed: ");
}

void dealloc_classify_alphas_on_GPU(){
    cudaFree(devAlphaC);
    CUT_CHECK_ERROR("Dealloc classify alphas on GPU failed: ");
}

void dealloc_train_result_on_GPU(){
    cudaFree(devResultT);
    CUT_CHECK_ERROR("Dealloc result on GPU failed: ");
}

void dealloc_classify_result_on_GPU(){
    cudaFree(devResultC);
    CUT_CHECK_ERROR("Dealloc result on GPU failed: ");
}

void dealloc_support_vectors_on_GPU(){
    cudaFree(devSV);
    CUT_CHECK_ERROR("Dealloc alphas on GPU failed: ");
}

int storeModel(int kernel_type, float gamma, float coef0,
               float degree, float* alpha, float* labels,
               float* data, int nPoints,
               int nDimension, float epsilon,
               float** support_vectors,
               float** out_alphas) { 

    int nSV = 0;
    int pSV = 0;
    for(int i = 0; i < nPoints; i++) {
        if (alpha[i] > epsilon) {
            if (labels[i] > 0) {
                pSV++;
            } else {
                nSV++;
            }
        }
    }

    int total_sv = pSV + nSV;
    float* local_sv = (float*)malloc(total_sv * nDimension * sizeof(float));
    float* local_alphas = (float*)malloc(total_sv * sizeof(float));

    int index = 0;
    for (int i = 0; i < nPoints; i++) {
        if (alpha[i] > epsilon) {
           local_alphas[index] =  labels[i] * alpha[i];
           for (int j = 0; j < nDimension; j++) {
               local_sv[j*total_sv + index] =  data[j * nPoints + i];
           }
           index++;
        }
    }

    *(support_vectors) = local_sv; 
    *(out_alphas) = local_alphas; 

    return total_sv;
}
