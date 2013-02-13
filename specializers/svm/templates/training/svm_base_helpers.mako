#define intDivideRoundUp(a, b) (a%b!=0)?(a/b+1):(a/b)

enum KernelType {
  LINEAR,
  POLYNOMIAL,
  GAUSSIAN,
  SIGMOID
};

// ============= CPU Data Allocation Functions ================
// CPU data structures
// Need to be linked to Python
float *data;
float *labels;
float *transposedData;
float *alphaT;
float *alphaC;
float *train_result;
float *classify_result;
float* support_vectors;

float *hostData;
bool hostDataAlloced;
bool transposedDataAlloced;
int hostPitchInFloats;
size_t devDataPitch;
int sizeOfCache;

void alloc_point_data_on_CPU(PyObject *input_data) {
  data = ((float*)PyArray_DATA(input_data));
}

void alloc_labels_on_CPU(PyObject *input_labels) {
  labels = ((float*)PyArray_DATA(input_labels));
}

void alloc_train_alphas_on_CPU(PyObject *input_alphas) {
  alphaT = ((float*)PyArray_DATA(input_alphas));
}

void alloc_classify_alphas_on_CPU(PyObject *input_alphas) {
  alphaC = ((float*)PyArray_DATA(input_alphas));
}

void alloc_train_result_on_CPU(PyObject *input_result) {
  train_result = ((float*)PyArray_DATA(input_result));
}

void alloc_classify_result_on_CPU(PyObject *input_result) {
  classify_result = ((float*)PyArray_DATA(input_result));
}

void alloc_support_vectors_on_CPU(PyObject *sv) {
  support_vectors = ((float*)PyArray_DATA(sv));
}
