// ================= FRAMEWORK =================
#define VERSION 0.1

#define BLOCKSIZE 512 
#define IMUL(a, b) __mul24(a, b)

#ifdef __DEVICE_EMULATION__
#define SYNC __syncthreads()
#else
#define SYNC 
#endif

#define REDUCE0  0x00000001
#define REDUCE1  0x00000002
#define REDUCE2  0x00000004
#define REDUCE3  0x00000008
#define REDUCE4  0x00000010
#define REDUCE5  0x00000020
#define REDUCE6  0x00000040
#define REDUCE7  0x00000080
#define REDUCE8  0x00000100
#define REDUCE9  0x00000200
#define REDUCE10 0x00000400
#define REDUCE11 0x00000800
#define REDUCE12 0x00001000
#define REDUCE13 0x00002000
#define REDUCE14 0x00004000
#define REDUCE15 0x00008000
#define REDUCE16 0x00010000
#define REDUCE17 0x00020000
#define REDUCE18 0x00040000
#define REDUCE19 0x00080000
#define REDUCE20 0x00100000
#define REDUCE21 0x00200000
#define REDUCE22 0x00400000
#define REDUCE23 0x00800000
#define REDUCE24 0x01000000
#define REDUCE25 0x02000000
#define REDUCE26 0x04000000
#define REDUCE27 0x08000000
#define REDUCE28 0x10000000
#define REDUCE29 0x20000000
#define REDUCE30 0x40000000
#define REDUCE31 0x80000000
#define NOREDUCE 0x00000000

#define INFTY  __int_as_float(0x7f000000)
#define NINFTY __int_as_float(0xff000000)

// ====================== DEVICE SELECT ====================
void chooseLargestGPU(bool verbose) {
    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);

    if (verbose) {
        printf("Total CUDA Devices Found: %d\n", cudaDeviceCount);
    }

    int cudaDevice = 0;
    int maxSps = 0;
    struct cudaDeviceProp dp;

    for (int i = 0; i < cudaDeviceCount; i++) {
        cudaGetDeviceProperties(&dp, i);
        if (dp.multiProcessorCount > maxSps) {
            maxSps = dp.multiProcessorCount;
            printf("device %d, max threads = %d\n", i, dp.maxThreadsPerBlock);
            cudaDevice = i;
        }
    }
    cudaGetDeviceProperties(&dp, cudaDevice);

    if (verbose) {
        printf("Using cuda device %i: %s\n", cudaDevice, dp.name);
    }

    cudaSetDevice(cudaDevice);
}

// ====================== MEMORY ROUTINES =================== 

__device__ void coopExtractRowVector(float* data, int dataPitch,
                                     int index, int dimension,
                                     float* destination) {
    float* xiRowPtr = data + (index * dataPitch) + threadIdx.x;
    for(int currentDim = threadIdx.x;
        currentDim < min(1500,dimension);
        currentDim += blockDim.x) {
        destination[currentDim] = *xiRowPtr;
        xiRowPtr += blockDim.x;
    }
}

// ==================== REDUCE OPERATORS ===============
inline __device__ void argMax(int mapAIndex, float mapAValue,
                              int mapBIndex, float mapBValue,
                              int* reduceIndex, float* reduceValue) {
    if (mapBValue > mapAValue) {
      *reduceIndex = mapBIndex;
      *reduceValue = mapBValue;
    } else {
      *reduceIndex = mapAIndex;
      *reduceValue = mapAValue;
    }
}

inline __device__ void argMin(int mapAIndex, float mapAValue,
                              int mapBIndex, float mapBValue,
                              int* reduceIndex, float* reduceValue) {
    if (mapBValue < mapAValue) {
        *reduceIndex = mapBIndex;
        *reduceValue = mapBValue;
    } else {
        *reduceIndex = mapAIndex;
        *reduceValue = mapAValue;
    }
}

template<typename T>
__device__ void maxOperator(T a, T b, T* result) {
    *result = (a > b) ? a : b;
}

// ================== REDUCE =================
template<int stepSize>
__device__ void sumStep(float* temps) {
    if (threadIdx.x < stepSize) {
        temps[threadIdx.x] += temps[threadIdx.x + stepSize];
    }
    if (stepSize >= 32) {
        __syncthreads();
    }
}

inline __device__ void sumReduce(float* temps) {
    if (256 < BLOCKSIZE) sumStep<256>(temps);
    if (128 < BLOCKSIZE) sumStep<128>(temps);
    if ( 64 < BLOCKSIZE) sumStep< 64>(temps);
    if ( 32 < BLOCKSIZE) sumStep< 32>(temps);
    if ( 16 < BLOCKSIZE) sumStep< 16>(temps);
    if (  8 < BLOCKSIZE) sumStep<  8>(temps);
    if (  4 < BLOCKSIZE) sumStep<  4>(temps);
    if (  2 < BLOCKSIZE) sumStep<  2>(temps);
    if (  1 < BLOCKSIZE) sumStep<  1>(temps);
}

template<int stepSize>
__device__ void maxStep(float* temps) {
    if (threadIdx.x < stepSize) {
        maxOperator(temps[threadIdx.x], temps[threadIdx.x + stepSize],
                    temps + threadIdx.x);
    }
    if (stepSize >= 32) {
        __syncthreads();
    }
}

inline __device__ void maxReduce(float* temps) {
    if (256 < BLOCKSIZE) maxStep<256>(temps);
    if (128 < BLOCKSIZE) maxStep<128>(temps);
    if ( 64 < BLOCKSIZE) maxStep< 64>(temps);
    if ( 32 < BLOCKSIZE) maxStep< 32>(temps);
    if ( 16 < BLOCKSIZE) maxStep< 16>(temps);
    if (  8 < BLOCKSIZE) maxStep<  8>(temps);
    if (  4 < BLOCKSIZE) maxStep<  4>(temps);
    if (  2 < BLOCKSIZE) maxStep<  2>(temps);
    if (  1 < BLOCKSIZE) maxStep<  1>(temps);
}

template<int stepSize>
__device__ void argminStep(float* values, int* indices) {
    if (threadIdx.x < stepSize) {
        int compOffset = threadIdx.x + stepSize;
        argMin(indices[threadIdx.x], values[threadIdx.x],
               indices[compOffset], values[compOffset],
               indices + threadIdx.x, values + threadIdx.x);
    }
    if (stepSize >= 32) {
        __syncthreads();
    }
}

inline __device__ void argminReduce(float* values, int* indices) {
    if (256 < BLOCKSIZE) argminStep<256>(values, indices);
    if (128 < BLOCKSIZE) argminStep<128>(values, indices);
    if ( 64 < BLOCKSIZE) argminStep< 64>(values, indices);
    if ( 32 < BLOCKSIZE) argminStep< 32>(values, indices);
    if ( 16 < BLOCKSIZE) argminStep< 16>(values, indices);
    if (  8 < BLOCKSIZE) argminStep<  8>(values, indices);
    if (  4 < BLOCKSIZE) argminStep<  4>(values, indices);
    if (  2 < BLOCKSIZE) argminStep<  2>(values, indices);
    if (  1 < BLOCKSIZE) argminStep<  1>(values, indices);
}

template<int stepSize>
__device__ void argmaxStep(float* values, int* indices) {
    if (threadIdx.x < stepSize) {
        int compOffset = threadIdx.x + stepSize;
        argMax(indices[threadIdx.x], values[threadIdx.x],
               indices[compOffset], values[compOffset],
               indices + threadIdx.x, values + threadIdx.x);
    }
    if (stepSize >= 32) {
        __syncthreads();
    }
}

inline __device__ void argmaxReduce(float* values, int* indices) {
    if (256 < BLOCKSIZE) argmaxStep<256>(values, indices);
    if (128 < BLOCKSIZE) argmaxStep<128>(values, indices);
    if ( 64 < BLOCKSIZE) argmaxStep< 64>(values, indices);
    if ( 32 < BLOCKSIZE) argmaxStep< 32>(values, indices);
    if ( 16 < BLOCKSIZE) argmaxStep< 16>(values, indices);
    if (  8 < BLOCKSIZE) argmaxStep<  8>(values, indices);
    if (  4 < BLOCKSIZE) argmaxStep<  4>(values, indices);
    if (  2 < BLOCKSIZE) argmaxStep<  2>(values, indices);
    if (  1 < BLOCKSIZE) argmaxStep<  1>(values, indices);
}
