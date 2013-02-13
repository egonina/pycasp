#include <math.h>
#include <float.h>

#  define CUT_CHECK_ERROR(errorMessage) {cudaError_t err = cudaGetLastError(); if( cudaSuccess != err) { fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n", errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );exit(EXIT_FAILURE); } }

#  define CUDA_SAFE_CALL_NO_SYNC( call) {cudaError err = call; if( cudaSuccess != err) {fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString( err) ); exit(EXIT_FAILURE);  } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);            

enum KernelType {
  LINEAR,
  POLYNOMIAL,
  GAUSSIAN,
  SIGMOID
};

// ===================== SVM KERNELS ====================
struct Linear {
    static __device__ __host__ float selfKernel(float* pointerA,
                                                int pitchA,
                                                float* pointerAEnd,
                                                float parameterA,
                                                float parameterB,
                                                float parameterC) {
        float accumulant = 0.0f;
        do {
            float value = *pointerA;
            accumulant += value * value;
            pointerA += pitchA;
        } while (pointerA < pointerAEnd);

        return accumulant;
    }
    
    static __device__ __host__ float kernel(float* pointerA,
                                            int pitchA,
                                            float* pointerAEnd,
                                            float* pointerB,
                                            int pitchB,
                                            float parameterA,
                                            float parameterB,
                                            float parameterC) {
        float accumulant = 0.0f;
        do {
            accumulant += (*pointerA) * (*pointerB);
            pointerA += pitchA;
            pointerB += pitchB;
        } while (pointerA < pointerAEnd);

        return accumulant;
    }
    
    static __device__ void parallelKernel(float* pointerA,
                                          float* pointerAEnd,
                                          float* pointerB,
                                          float* sharedTemps,
                                          float parameterA,
                                          float parameterB,
                                          float parameterC) {
      pointerA += threadIdx.x;
      pointerB += threadIdx.x;
      sharedTemps[threadIdx.x] = 0.0f;
      while (pointerA < pointerAEnd) {
        sharedTemps[threadIdx.x] += (*pointerA) * (*pointerB);
        pointerA += blockDim.x;
        pointerB += blockDim.x;
      }
      __syncthreads();
    
      sumReduce(sharedTemps);
    }
    
    //This function assumes we're doing two kernel evaluations at once:
    //Phi1(a, b) and Phi2(a, c)
    //b and c are already in shared memory, so we don't care about minimizing
    //their memory accesses, but a is in global memory
    //So we have to worry about not accessing a twice
    static __device__ __host__ void dualKernel(float* pointerA, int pitchA,
                                               float* pointerAEnd,
                                               float* pointerB, int pitchB,
                                               float* pointerC, int pitchC,
                                               float parameterA,
                                               float parameterB,
                                               float parameterC,
                                               float& phi1,
                                               float& phi2) {
      float accumulant1 = 0.0f;
      float accumulant2 = 0.0f;
      int counter= 0;
      do {
        float xa = *pointerA;
        accumulant1 += xa * (*pointerB);
        accumulant2 += xa * (*pointerC);
        pointerA += pitchA;
        pointerB += pitchB;
        pointerC += pitchC;
        counter++;
      } while (pointerA < pointerAEnd);
      phi1 = accumulant1;
      phi2 = accumulant2;
    }
};

struct Polynomial {
    static __device__ __host__ float selfKernel(float* pointerA, int pitchA,
                                                float* pointerAEnd,
                                                float a, float r, float d) {
        float accumulant = 0.0f;
        do {
            float value = *pointerA;
            accumulant += value * value;
            pointerA += pitchA;
        } while (pointerA < pointerAEnd);
        accumulant = accumulant * a + r;
        float result = accumulant;
        for (float degree = 2.0f; degree <= d; degree = degree + 1.0f) {
            result *= accumulant;
        }
        return result;
    }
    
    static __device__ __host__ float kernel(float* pointerA, int pitchA,
                                            float* pointerAEnd,
                                            float* pointerB, int pitchB,
                                            float a, float r, float d) {
        float accumulant = 0.0f;
        do {
            accumulant += (*pointerA) * (*pointerB);
            pointerA += pitchA;
            pointerB += pitchB;
        } while (pointerA < pointerAEnd);
        accumulant = accumulant * a + r;
        float result = accumulant;
        for (float degree = 2.0f; degree <= d; degree = degree + 1.0f) {
            result *= accumulant;
        }
        return result;
    }
    
    static __device__ void parallelKernel(float* pointerA, float* pointerAEnd,
                                          float* pointerB, float* sharedTemps,
                                          float a, float r, float d) {
        pointerA += threadIdx.x;
        pointerB += threadIdx.x;
        sharedTemps[threadIdx.x] = 0.0f;
        while (pointerA < pointerAEnd) {
            sharedTemps[threadIdx.x] += (*pointerA) * (*pointerB);
            pointerA += blockDim.x;
            pointerB += blockDim.x;
        }
        __syncthreads();
    
        sumReduce(sharedTemps);
        if (threadIdx.x == 0) {
            float accumulant = sharedTemps[0] * a + r;
           
            float result = accumulant;
            for (float degree = 2.0f; degree <= d; degree = degree + 1.0f) {
                result *= accumulant;
            }
            sharedTemps[0] = result;
        }
    }

    //This function assumes we're doing two kernel evaluations at once:
    //Phi1(a, b) and Phi2(a, c)
    //b and c are already in shared memory, so we don't care about minimizing
    //their memory accesses, but a is in global memory
    //So we have to worry about not accessing a twice
    static __device__ __host__ void dualKernel(float* pointerA, int pitchA,
                                               float* pointerAEnd,
                                               float* pointerB, int pitchB,
                                               float* pointerC, int pitchC,
                                               float a, float r, float d,
                                               float& phi1, float& phi2) {
        float accumulant1 = 0.0f;
        float accumulant2 = 0.0f;
        do {
            float xa = *pointerA;
            accumulant1 += xa * (*pointerB);
            accumulant2 += xa * (*pointerC);
            pointerA += pitchA;
            pointerB += pitchB;
            pointerC += pitchC;
        } while (pointerA < pointerAEnd);
        
        accumulant1 = accumulant1 * a + r;
        float result = accumulant1;
        for (float degree = 2.0f; degree <= d; degree = degree + 1.0f) {
            result *= accumulant1;
        }
        phi1 = result;
    
        accumulant2 = accumulant2 * a + r;
        result = accumulant2;
        for (float degree = 2.0f; degree <= d; degree = degree + 1.0f) {
            result *= accumulant2;
        }
        phi2 = result;
    }
};

struct Gaussian {
    static __device__ __host__ float selfKernel(float* pointerA, int pitchA,
                                                float* pointerAEnd, float ngamma,
                                                float parameterB, float parameterC) {
        return 1.0f;
    }
  
    static __device__ __host__ float kernel(float* pointerA, int pitchA,
                                            float* pointerAEnd,
                                            float* pointerB, int pitchB,
                                            float ngamma, float parameterB,
                                            float parameterC) {
        float accumulant = 0.0f;
        do {
            float diff = *pointerA - *pointerB;
            accumulant += diff * diff;
            pointerA += pitchA;
            pointerB += pitchB;
        } while (pointerA < pointerAEnd);
        return exp(ngamma * accumulant);
    }
  
    static __device__ void parallelKernel(float* pointerA, float* pointerAEnd,
                                          float* pointerB, float* sharedTemps,
                                          float ngamma, float parameterB,
                                          float parameterC) {
        pointerA += threadIdx.x;
        pointerB += threadIdx.x;
        sharedTemps[threadIdx.x] = 0.0f;
        while (pointerA < pointerAEnd) {
            float diff = (*pointerA) - (*pointerB);
            sharedTemps[threadIdx.x] += diff * diff;
            pointerA += blockDim.x;
            pointerB += blockDim.x;
        }
        __syncthreads();
  
        sumReduce(sharedTemps);
        if (threadIdx.x == 0) {
            sharedTemps[0] = exp(sharedTemps[0] * ngamma);
        }
    }
    
    //This function assumes we're doing two kernel evaluations at once:
    //Phi1(a, b) and Phi2(a, c)
    //b and c are already in shared memory, so we don't care about minimizing
    //their memory accesses, but a is in global memory
    //So we have to worry about not accessing a twice
    static __device__ __host__ void dualKernel(float* pointerA, int pitchA,
                                               float* pointerAEnd,
                                               float* pointerB, int pitchB,
                                               float* pointerC, int pitchC,
                                               float ngamma, float parameterB,
                                               float parameterC,
                                               float& phi1, float& phi2) {
        float accumulant1 = 0.0f;
        float accumulant2 = 0.0f;
        do {
            float xa = *pointerA;
            float diff = xa - (*pointerB);
            accumulant1 += diff * diff;
            diff = xa - (*pointerC);
            accumulant2 += diff * diff;
            pointerA += pitchA;
            pointerB += pitchB;
            pointerC += pitchC;
        } while (pointerA < pointerAEnd);
        phi1 = exp(ngamma * accumulant1);
        phi2 = exp(ngamma * accumulant2);
    }
    
};

struct Sigmoid {
    static __device__ __host__ float selfKernel(float* pointerA, int pitchA,
                                                float* pointerAEnd,
                                                float a, float r, float parameterC) {
        float accumulant = 0.0f;
        do {
            float value = *pointerA;
            accumulant += value * value;
            pointerA += pitchA;
        } while (pointerA < pointerAEnd);
        accumulant = accumulant * a + r;
        return tanh(accumulant);
    }

    static __device__ __host__ float kernel(float* pointerA, int pitchA,
                                            float* pointerAEnd,
                                            float* pointerB, int pitchB,
                                            float a, float r, float parameterC) {
        float accumulant = 0.0f;
        do {
            accumulant += (*pointerA) * (*pointerB);
            pointerA += pitchA;
            pointerB += pitchB;
        } while (pointerA < pointerAEnd);
        accumulant = accumulant * a + r;
        return tanh(accumulant);
    }

    static __device__ void parallelKernel(float* pointerA, float* pointerAEnd,
                                          float* pointerB, float* sharedTemps,
                                          float a, float r, float parameterC) {
        pointerA += threadIdx.x;
        pointerB += threadIdx.x;
        sharedTemps[threadIdx.x] = 0.0f;
        while (pointerA < pointerAEnd) {
            sharedTemps[threadIdx.x] += (*pointerA) * (*pointerB);
            pointerA += blockDim.x;
            pointerB += blockDim.x;
        }
        __syncthreads();

        sumReduce(sharedTemps);
        if (threadIdx.x == 0) {
            float accumulant = sharedTemps[0];
            sharedTemps[0] = tanh(accumulant);
        }
    }

    //This function assumes we're doing two kernel evaluations at once:
    //Phi1(a, b) and Phi2(a, c)
    //b and c are already in shared memory, so we don't care about minimizing
    //their memory accesses, but a is in global memory
    //So we have to worry about not accessing a twice
    static __device__ __host__ void dualKernel(float* pointerA, int pitchA,
                                               float* pointerAEnd,
                                               float* pointerB, int pitchB,
                                               float* pointerC, int pitchC,
                                               float a, float r, float parameterC,
                                               float& phi1, float& phi2) {
        float accumulant1 = 0.0f;
        float accumulant2 = 0.0f;
        do {
            float xa = *pointerA;
            accumulant1 += xa * (*pointerB);
            accumulant2 += xa * (*pointerC);
            pointerA += pitchA;
            pointerB += pitchB;
            pointerC += pitchC;
        } while (pointerA < pointerAEnd);
        accumulant1 = accumulant1 * a + r;
        phi1= tanh(accumulant1);
        accumulant2 = accumulant2 * a + r;
        phi2= tanh(accumulant2);
    }
};

// ===================== TRAINIG KERNELS ====================

// =================== INITIALIZE ===================
template<class Kernel>
__global__ void initializeArrays(float* devData,
                                 int devDataPitchInFloats,
                                 int nPoints,
                                 int nDimension,
                                 float parameterA,
                                 float parameterB,
                                 float parameterC,
                                 float* devKernelDiag,
                                 float* devAlphaT,
                                 float* devF,
                                 float* devLabels) { 
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    
    int index = bx * blockDim.x + tx;
    
    if (index < nPoints) {
        devKernelDiag[index] = Kernel::selfKernel(devData + index,
                                                  devDataPitchInFloats,
                                                  devData + (nDimension * devDataPitchInFloats),
                                                  parameterA, parameterB, parameterC);
    	devF[index] = -devLabels[index];
    	devAlphaT[index] = 0;
    }
}

template<class Kernel>
__global__ void takeFirstStep(void* devResultT, float* devKernelDiag,
                              float* devData, int devDataPitchInFloats,
                              float* devAlphaT, float cost, int nDimension,
                              int iLow, int iHigh, float parameterA,
                              float parameterB, float parameterC) { 
                                     
    float eta = devKernelDiag[iHigh] + devKernelDiag[iLow];
    float* pointerA = devData + iHigh;
    float* pointerB = devData + iLow;
    float* pointerAEnd = devData + IMUL(nDimension, devDataPitchInFloats);
    float phiAB = Kernel::kernel(pointerA, devDataPitchInFloats, pointerAEnd, pointerB, devDataPitchInFloats, parameterA, parameterB, parameterC);
	
    eta = eta - 2*phiAB;
    //For the first step, we know alpha1Old == alpha2Old == 0, and we know sign == -1
    //labels[iLow] = -1
    //labels[iHigh] = 1
    //float sign = -1;
    
    //And we know eta > 0
    float alpha2New = 2/eta; //Just boil down the algebra
    if (alpha2New > cost) {
    	alpha2New = cost;
    }
    //alpha1New == alpha2New for the first step
    
    devAlphaT[iLow] = alpha2New;
    devAlphaT[iHigh] = alpha2New;
    
    *((float*)devResultT + 0) = 0.0;
    *((float*)devResultT + 1) = 0.0;
    *((float*)devResultT + 2) = 1.0;
    *((float*)devResultT + 3) = -1.0;
    *((float*)devResultT + 6) = alpha2New;
    *((float*)devResultT + 7) = alpha2New;
}

// =============== FIRSTORDER =============

template<bool iLowCompute, bool iHighCompute, class Kernel>
__global__ void	firstOrderPhaseOne(float* devData, int devDataPitchInFloats,
                                   float* devTransposedData,
                                   int devTransposedDataPitchInFloats,
                                   float* devLabels, int nPoints, int nDimension,
                                   float epsilon, float cEpsilon,
                                   float* devAlphaT, float* devF,
                                   float alpha1Diff, float alpha2Diff,
                                   int iLow, int iHigh, float parameterA,
                                   float parameterB, float parameterC,
                                   float* devCache, int devCachePitchInFloats,
                                   int iLowCacheIndex, int iHighCacheIndex,
                                   int* devLocalIndicesRL, int* devLocalIndicesRH,
                                   float* devLocalFsRL, float* devLocalFsRH) {
    extern __shared__ float xIHigh[];
    float* xILow;
    __shared__ int tempLocalIndices[BLOCKSIZE];
    __shared__ float tempLocalFs[BLOCKSIZE];
    
    if (iHighCompute) {
    	xILow = &xIHigh[nDimension];
    } else {
    	xILow = xIHigh;
    }
    
    if (iHighCompute) {
      //Load xIHigh into shared memory
      coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats,
                           iHigh, nDimension, xIHigh);
    }

    if (iLowCompute) {
      //Load xILow into shared memory
      coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats,
                           iLow, nDimension, xILow);
    }
    __syncthreads();
      
    int globalIndex = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

    float alpha;
    float f;
    float label;
    int reduceFlag;
    
    if (globalIndex < nPoints) {
        alpha = devAlphaT[globalIndex];
        
        f = devF[globalIndex];
        label = devLabels[globalIndex];
        
        
        if (alpha > epsilon) {
            if (alpha < cEpsilon) {
           	    reduceFlag = REDUCE0 | REDUCE1; //Unbound support vector (I0)
            } else {
                if (label > 0) {
               	    reduceFlag = REDUCE0; //Bound positive support vector (I3)
                } else {
                    reduceFlag = REDUCE1; //Bound negative support vector (I2)
                }
            }
        } else {
            if (label > 0) {
                reduceFlag = REDUCE1; //Positive nonsupport vector (I1)
            } else {
                reduceFlag = REDUCE0; //Negative nonsupport vector (I4)
            }
        }
    } else {
        reduceFlag = NOREDUCE;
    }
    
    float highKernel = 0;
    float lowKernel = 0;
    if (reduceFlag > 0) {
        if (!iHighCompute) {
            highKernel = devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex];

        }
        if (!iLowCompute) {
            lowKernel = devCache[(devCachePitchInFloats * iLowCacheIndex) + globalIndex];
        }
        if(nDimension < 1500)
        {
            if (iHighCompute && iLowCompute) {
              Kernel::dualKernel(devData + globalIndex,
                                 devDataPitchInFloats,
                                 devData + globalIndex + (devDataPitchInFloats * nDimension),
                                 xIHigh, 1, xILow, 1, parameterA, parameterB, parameterC,
                                 highKernel, lowKernel);
            } else if (iHighCompute) {
                highKernel = Kernel::kernel(devData + globalIndex,
                                            devDataPitchInFloats,
                                            devData + globalIndex + (devDataPitchInFloats * nDimension),
                                            xIHigh, 1, parameterA, parameterB, parameterC);
            } else if (iLowCompute) {
                lowKernel = Kernel::kernel(devData + globalIndex,
                                           devDataPitchInFloats,
                                           devData + globalIndex + (devDataPitchInFloats * nDimension),
                                           xILow, 1, parameterA, parameterB, parameterC);
            }
        }
        else
        {
             if (iHighCompute && iLowCompute) {
               Kernel::dualKernel(devData + globalIndex,
                                  devDataPitchInFloats,
                                   devData + globalIndex + (devDataPitchInFloats * nDimension), 
                                   devTransposedData + iHigh*devTransposedDataPitchInFloats, 1, 
                                   devTransposedData + iLow*devTransposedDataPitchInFloats, 1,
                                   parameterA, parameterB, parameterC, highKernel, lowKernel);
              } else if (iHighCompute) {
                  highKernel = Kernel::kernel(devData + globalIndex,
                                            devDataPitchInFloats,
                                            devData + globalIndex + (devDataPitchInFloats * nDimension), 
                                            devTransposedData + iHigh*devTransposedDataPitchInFloats, 1, 
                                            parameterA, parameterB, parameterC);
              } else if (iLowCompute) {
                  lowKernel = Kernel::kernel(devData + globalIndex,
                                           devDataPitchInFloats,
                                           devData + globalIndex + (devDataPitchInFloats * nDimension), 
                                           devTransposedData + iLow*devTransposedDataPitchInFloats, 1, 
                                           parameterA, parameterB, parameterC);
             }
        }
        
        f = f + alpha1Diff * highKernel;
        f = f + alpha2Diff * lowKernel;
        
        if (iLowCompute) {
            devCache[(devCachePitchInFloats * iLowCacheIndex) + globalIndex] = lowKernel;
        }
        if (iHighCompute) {
            devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex] = highKernel;
        }
      	devF[globalIndex] = f;
    }
    __syncthreads();


    if ((reduceFlag & REDUCE0) == 0) {
        tempLocalFs[threadIdx.x] = -FLT_MAX; //Ignore me
    } else {
        tempLocalFs[threadIdx.x] = f;
        tempLocalIndices[threadIdx.x] = globalIndex;
    }
    __syncthreads();
    
    argmaxReduce(tempLocalFs, tempLocalIndices);
    
    
    if (threadIdx.x == 0) {
      	devLocalIndicesRL[blockIdx.x] = tempLocalIndices[0];
      	devLocalFsRL[blockIdx.x] = tempLocalFs[0];
    }
 
    __syncthreads();

    if ((reduceFlag & REDUCE1) == 0) {
        tempLocalFs[threadIdx.x] = FLT_MAX; //Ignore me
    } else {
        tempLocalFs[threadIdx.x] = f;
        tempLocalIndices[threadIdx.x] = globalIndex;
    }
    __syncthreads();
     
    argminReduce(tempLocalFs, tempLocalIndices);

    if (threadIdx.x == 0) {
        devLocalIndicesRH[blockIdx.x] = tempLocalIndices[0];
        devLocalFsRH[blockIdx.x] = tempLocalFs[0];
    }
}

template<class Kernel>
__global__ void firstOrderPhaseTwo(float* devData, int devDataPitchInFloats,
                                   float* devTransposedData,
                                   int devTransposedDataPitchInFloats,
                                   float* devLabels, float* devKernelDiag,
                                   float* devAlphaT, void* devResultT,
                                   float cost, int nDimension,
                                   float parameterA, float parameterB, float parameterC,
                                   int* devLocalIndicesRL, int* devLocalIndicesRH,
                                   float* devLocalFsRL, float* devLocalFsRH, int inputSize) {
    __shared__ int tempIndices[BLOCKSIZE];
    __shared__ float tempFs[BLOCKSIZE];
    
    //Load elements
    if (threadIdx.x < inputSize) {
        tempIndices[threadIdx.x] = devLocalIndicesRH[threadIdx.x];
        tempFs[threadIdx.x] = devLocalFsRH[threadIdx.x];
    } else {
        tempFs[threadIdx.x] = FLT_MAX;
    }

    if (inputSize > BLOCKSIZE) {
        for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
            argMin(tempIndices[threadIdx.x], tempFs[threadIdx.x],
                   devLocalIndicesRH[i], devLocalFsRH[i],
                   tempIndices + threadIdx.x, tempFs + threadIdx.x);
        }
    }
    __syncthreads();

    argminReduce(tempFs, tempIndices);
    int iHigh = tempIndices[0];
    float bHigh = tempFs[0];

    //Load elements
    if (threadIdx.x < inputSize) {
        tempIndices[threadIdx.x] = devLocalIndicesRL[threadIdx.x];
        tempFs[threadIdx.x] = devLocalFsRL[threadIdx.x];
    } else {
        tempFs[threadIdx.x] = -FLT_MAX;
    }

    if (inputSize > BLOCKSIZE) {
        for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
            argMax(tempIndices[threadIdx.x], tempFs[threadIdx.x],
                   devLocalIndicesRL[i], devLocalFsRL[i],
                   tempIndices + threadIdx.x, tempFs + threadIdx.x);
        }
    }
    __syncthreads();
    
    argmaxReduce(tempFs, tempIndices);
    
    int iLow = tempIndices[0];
    float bLow = tempFs[0];
    
    float* highPointer = devTransposedData + (iHigh * devTransposedDataPitchInFloats);
    float* lowPointer = devTransposedData + (iLow * devTransposedDataPitchInFloats);  
    
    Kernel::parallelKernel(highPointer, highPointer + nDimension,
                           lowPointer, tempFs,
                           parameterA, parameterB, parameterC);

    if (threadIdx.x == 0) {
        float eta = devKernelDiag[iHigh] + devKernelDiag[iLow];
          
        float kernelEval = tempFs[0];
        eta = eta - 2*kernelEval;
          
        float alpha1Old = devAlphaT[iHigh];
        float alpha2Old = devAlphaT[iLow];
        float alphaDiff = alpha2Old - alpha1Old;
        float lowLabel = devLabels[iLow];
        float sign = devLabels[iHigh] * lowLabel;
        float alpha2UpperBound;
        float alpha2LowerBound;
        if (sign < 0) {
            if (alphaDiff < 0) {
                alpha2LowerBound = 0;
                alpha2UpperBound = cost + alphaDiff;
            } else {
                alpha2LowerBound = alphaDiff;
                alpha2UpperBound = cost;
            }
        } else {
            float alphaSum = alpha2Old + alpha1Old;
            if (alphaSum < cost) {
                alpha2UpperBound = alphaSum;
                alpha2LowerBound = 0;
            } else {
                alpha2LowerBound = alphaSum - cost;
                alpha2UpperBound = cost;
        	}
        }
        float alpha2New;
        if (eta > 0) {
            alpha2New = alpha2Old + lowLabel*(bHigh - bLow)/eta;
            if (alpha2New < alpha2LowerBound) {
                alpha2New = alpha2LowerBound;
            } else if (alpha2New > alpha2UpperBound) {
            	alpha2New = alpha2UpperBound;
            }
        } else {
            float slope = lowLabel * (bHigh - bLow);
            float delta = slope * (alpha2UpperBound - alpha2LowerBound);
            if (delta > 0) {
                if (slope > 0) {
                    alpha2New = alpha2UpperBound;
                } else {
                    alpha2New = alpha2LowerBound;
                }
            } else {
                alpha2New = alpha2Old;
            }
        }

        float alpha2Diff = alpha2New - alpha2Old;
        float alpha1Diff = -sign*alpha2Diff;
        float alpha1New = alpha1Old + alpha1Diff;

        *((float*)devResultT + 0) = alpha2Old;
        *((float*)devResultT + 1) = alpha1Old;
        *((float*)devResultT + 2) = bLow;
        *((float*)devResultT + 3) = bHigh;
        devAlphaT[iLow] = alpha2New;
        devAlphaT[iHigh] = alpha1New;
        *((float*)devResultT + 4) = alpha2New;
        *((float*)devResultT + 5) = alpha1New;
        *((int*)devResultT + 6) = iLow;
        *((int*)devResultT + 7) = iHigh;
        *((float*)devResultT + 8) = eta;
        *((float*)devResultT + 9) = kernelEval;
    }
}

// ================= SECONDORDER =================

template<bool iLowCompute, bool iHighCompute, class Kernel>
__global__ void	secondOrderPhaseOne(float* devData, int devDataPitchInFloats,
                                    float* devTransposedData,
                                    int devTransposedDataPitchInFloats,
                                    float* devLabels, int nPoints, int nDimension,
                                    float epsilon, float cEpsilon, float* devAlphaT,
                                    float* devF, float alpha1Diff, float alpha2Diff,
                                    int iLow, int iHigh, float parameterA,
                                    float parameterB, float parameterC, float* devCache,
                                    int devCachePitchInFloats, int iLowCacheIndex,
                                    int iHighCacheIndex, int* devLocalIndicesRH,
                                    float* devLocalFsRH) {
  
    extern __shared__ float xIHigh[];
    float* xILow;
    __shared__ int tempLocalIndices[BLOCKSIZE];
    __shared__ float tempLocalFs[BLOCKSIZE];

    if (iHighCompute) {
        xILow = &xIHigh[nDimension];
    } else {
        xILow = xIHigh;
    }

    if (iHighCompute) {
        coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats,
                             iHigh, nDimension, xIHigh);
    }

    if (iLowCompute) {
        coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats,
                             iLow, nDimension, xILow);
    }
    __syncthreads();

    int globalIndex = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

    float alpha;
    float f;
    float label;
    int reduceFlag;

    if (globalIndex < nPoints) {
        alpha = devAlphaT[globalIndex];
        f = devF[globalIndex];
        label = devLabels[globalIndex];
    }
    
    if ((globalIndex < nPoints) &&
        (((label > 0) && (alpha < cEpsilon)) ||
        ((label < 0) && (alpha > epsilon)))) {
        reduceFlag = REDUCE0;
    } else {
        reduceFlag = NOREDUCE;
    }

    if (globalIndex < nPoints) {
        float highKernel;
        float lowKernel;
        if (iHighCompute) {
            highKernel = 0;
        } else {
            highKernel = devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex];
        }
        if (iLowCompute) {
            lowKernel = 0;
        } else {
            lowKernel = devCache[(devCachePitchInFloats * iLowCacheIndex) + globalIndex];
        }
        if(nDimension < 1500) {
            if (iHighCompute && iLowCompute) {
                Kernel::dualKernel(devData + globalIndex, devDataPitchInFloats,
                                   devData + globalIndex + (devDataPitchInFloats * nDimension),
                                   xIHigh, 1, xILow, 1, parameterA, parameterB, parameterC,
                                   highKernel, lowKernel);
            } else if (iHighCompute) {
                highKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats,
                                            devData + globalIndex + (devDataPitchInFloats * nDimension),
                                            xIHigh, 1, parameterA, parameterB, parameterC);
            } else if (iLowCompute) {
                lowKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats,
                                           devData + globalIndex + (devDataPitchInFloats * nDimension),
                                           xILow, 1, parameterA, parameterB, parameterC);
            }
        }
        else {    
          if (iHighCompute && iLowCompute) {
              Kernel::dualKernel(devData + globalIndex, devDataPitchInFloats,
                                 devData + globalIndex + (devDataPitchInFloats * nDimension),
                                 devTransposedData + iHigh*devTransposedDataPitchInFloats, 1,
                                 devTransposedData + iLow*devTransposedDataPitchInFloats, 1,
                                 parameterA, parameterB, parameterC, highKernel, lowKernel);
          } else if (iHighCompute) {
              highKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats,
                                          devData + globalIndex + (devDataPitchInFloats * nDimension),  
                                          devTransposedData + iHigh*devTransposedDataPitchInFloats, 1,
                                          parameterA, parameterB, parameterC);
          } else if (iLowCompute) {
              lowKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats,
                                         devData + globalIndex + (devDataPitchInFloats * nDimension),  
                                         devTransposedData + iLow*devTransposedDataPitchInFloats, 1,
                                         parameterA, parameterB, parameterC);
          }
        }
      		
        f = f + alpha1Diff * highKernel;
        f = f + alpha2Diff * lowKernel;
        	
        if (iLowCompute) {
            devCache[(devCachePitchInFloats * iLowCacheIndex) + globalIndex] = lowKernel;
        }
        if (iHighCompute) {
            devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex] = highKernel;
        }
        devF[globalIndex] = f;
    }
    __syncthreads();

    if ((reduceFlag & REDUCE0) == 0) {
        tempLocalFs[threadIdx.x] = FLT_MAX; //Ignore me
    } else {
        tempLocalFs[threadIdx.x] = f;
        tempLocalIndices[threadIdx.x] = globalIndex;
    }
    __syncthreads(); 
    argminReduce(tempLocalFs, tempLocalIndices);
    if (threadIdx.x == 0) {
        devLocalIndicesRH[blockIdx.x] = tempLocalIndices[0];
        devLocalFsRH[blockIdx.x] = tempLocalFs[0];
    }
}

__global__ void secondOrderPhaseTwo(void* devResultT, int* devLocalIndicesRH,
                                    float* devLocalFsRH, int inputSize) {
    __shared__ int tempIndices[BLOCKSIZE];
    __shared__ float tempFs[BLOCKSIZE];
  
    //Load elements
    if (threadIdx.x < inputSize) {
       tempIndices[threadIdx.x] = devLocalIndicesRH[threadIdx.x];
       tempFs[threadIdx.x] = devLocalFsRH[threadIdx.x];
    } else {
       tempFs[threadIdx.x] = FLT_MAX;
    }
  
    if (inputSize > BLOCKSIZE) {
        for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
          argMin(tempIndices[threadIdx.x], tempFs[threadIdx.x],
                 devLocalIndicesRH[i], devLocalFsRH[i],
                 tempIndices + threadIdx.x, tempFs + threadIdx.x);
        }
    }
    __syncthreads();
    argminReduce(tempFs, tempIndices);
    int iHigh = tempIndices[0];
    float bHigh = tempFs[0];
  
    if (threadIdx.x == 0) {
        *((float*)devResultT + 3) = bHigh;
        *((int*)devResultT + 7) = iHigh;
    }
}

template <bool iHighCompute, class Kernel>
__global__ void secondOrderPhaseThree(float* devData, int devDataPitchInFloats,
                                      float* devTransposedData,
                                      int devTransposedDataPitchInFloats,
                                      float* devLabels, float* devKernelDiag,
                                      float epsilon, float cEpsilon, float* devAlphaT,
                                      float* devF, float bHigh, int iHigh, float* devCache,
                                      int devCachePitchInFloats, int iHighCacheIndex,
                                      int nDimension, int nPoints, float parameterA,
                                      float parameterB, float parameterC, float* devLocalFsRL,
                                      int* devLocalIndicesMaxObj, float* devLocalObjsMaxObj) {
    extern __shared__ float xIHigh[];
    __shared__ int tempIndices[BLOCKSIZE];
    __shared__ float tempValues[BLOCKSIZE];
    __shared__ float iHighSelfKernel;
      
    if (iHighCompute) {
      //Load xIHigh into shared memory
      coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats, iHigh, nDimension, xIHigh);
    }

    if (threadIdx.x == 0) {
        iHighSelfKernel = devKernelDiag[iHigh];
    }
    __syncthreads();

    int globalIndex = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

    float alpha;
    float f;
    float label;
    int reduceFlag;
    float obj;

    if (globalIndex < nPoints) {
        alpha = devAlphaT[globalIndex];

        f = devF[globalIndex];
        label = devLabels[globalIndex];
        
        float highKernel;
        if (iHighCompute) {
            highKernel = 0;
        } else {
           	highKernel = devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex];
        }

        if (iHighCompute) {
            if(nDimension <1500) {
                highKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats,
                                            devData + globalIndex + (devDataPitchInFloats * nDimension),
                                            xIHigh, 1, parameterA, parameterB, parameterC);
            } else {
                highKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats,
                                            devData + globalIndex + (devDataPitchInFloats * nDimension), 
                                            devTransposedData+iHigh*devTransposedDataPitchInFloats, 1,
                                            parameterA, parameterB, parameterC);
            }

            devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex] = highKernel;
        }
        
        float beta = bHigh - f;

        float kappa = iHighSelfKernel + devKernelDiag[globalIndex] - 2 * highKernel;
        
        if (kappa <= 0) {
            kappa = epsilon;
        }
        
        obj = beta * beta / kappa;
        if (((label > 0) && (alpha > epsilon)) ||
            ((label < 0) && (alpha < cEpsilon))) {
            if (beta <= epsilon) {
                reduceFlag = REDUCE1 | REDUCE0;
            } else {        
                reduceFlag = REDUCE0;
            }
        } else {
            reduceFlag = NOREDUCE;
        }
    } else {
      reduceFlag = NOREDUCE;
    }

    if ((reduceFlag & REDUCE0) == 0) {
        tempValues[threadIdx.x] = -FLT_MAX; //Ignore me
    } else {
        tempValues[threadIdx.x] = f;
    }
    
    __syncthreads();
    
    maxReduce(tempValues);
    if (threadIdx.x == 0) {
   	    devLocalFsRL[blockIdx.x] = tempValues[0];
    }

    if ((reduceFlag & REDUCE1) == 0) {
        tempValues[threadIdx.x] = -FLT_MAX; //Ignore me
        tempIndices[threadIdx.x] = 0;
    } else {
        tempValues[threadIdx.x] = obj;
        tempIndices[threadIdx.x] = globalIndex;
    }
    __syncthreads();
    argmaxReduce(tempValues, tempIndices);
    
    if (threadIdx.x == 0) {
        devLocalIndicesMaxObj[blockIdx.x] = tempIndices[0];
        devLocalObjsMaxObj[blockIdx.x] = tempValues[0];
    }
}

__global__ void secondOrderPhaseFour(float* devLabels, float* devKernelDiag,
                                     float* devF, float* devAlphaT,
                                     float cost, int iHigh, float bHigh,
                                     void* devResultT, float* devCache,
                                     int devCachePitchInFloats, int iHighCacheIndex,
                                     float* devLocalFsRL, int* devLocalIndicesMaxObj,
                                     float* devLocalObjsMaxObj, int inputSize,
                                     int iteration) {
    __shared__ int tempIndices[BLOCKSIZE];
    __shared__ float tempValues[BLOCKSIZE];
    
    if (threadIdx.x < inputSize) {
        tempIndices[threadIdx.x] = devLocalIndicesMaxObj[threadIdx.x];
        tempValues[threadIdx.x] = devLocalObjsMaxObj[threadIdx.x];
    } else {
        tempValues[threadIdx.x] = -FLT_MAX;
        tempIndices[threadIdx.x] = -1;
    }

    if (inputSize > BLOCKSIZE) {
        for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
            argMax(tempIndices[threadIdx.x], tempValues[threadIdx.x],
                   devLocalIndicesMaxObj[i], devLocalObjsMaxObj[i],
                   tempIndices + threadIdx.x, tempValues + threadIdx.x);
        }
    }
    __syncthreads();
    
    argmaxReduce(tempValues, tempIndices);

    __syncthreads();
    int iLow;
    if (threadIdx.x == 0) {
        iLow = tempIndices[0];
    }
 
    if (threadIdx.x < inputSize) {
        tempValues[threadIdx.x] = devLocalFsRL[threadIdx.x];
    } else {
        tempValues[threadIdx.x] = -FLT_MAX;
    }
    
    if (inputSize > BLOCKSIZE) {
        for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
          maxOperator(tempValues[threadIdx.x], devLocalFsRL[i],
                      tempValues + threadIdx.x);
        }
    }
    __syncthreads();
    maxReduce(tempValues);
    __syncthreads();

    float bLow;
    if (threadIdx.x == 0) {
        bLow = tempValues[0];
    }
    
    if (threadIdx.x == 0) {
        float eta = devKernelDiag[iHigh] + devKernelDiag[iLow];
     
        eta = eta - 2 * (*(devCache + (devCachePitchInFloats * iHighCacheIndex) + iLow));
        
        float alpha1Old = devAlphaT[iHigh];
        float alpha2Old = devAlphaT[iLow];
        float alphaDiff = alpha2Old - alpha1Old;
        float lowLabel = devLabels[iLow];
        float sign = devLabels[iHigh] * lowLabel;
        float alpha2UpperBound;
        float alpha2LowerBound;
        if (sign < 0) {
            if (alphaDiff < 0) {
                alpha2LowerBound = 0;
                alpha2UpperBound = cost + alphaDiff;
            } else {
                alpha2LowerBound = alphaDiff;
                alpha2UpperBound = cost;
            }
        } else {
            float alphaSum = alpha2Old + alpha1Old;
            if (alphaSum < cost) {
                alpha2UpperBound = alphaSum;
                alpha2LowerBound = 0;
            } else {
                alpha2LowerBound = alphaSum - cost;
                alpha2UpperBound = cost;
            }
        }
        float alpha2New;
        if (eta > 0) {
            alpha2New = alpha2Old + lowLabel*(devF[iHigh] - devF[iLow])/eta;
            if (alpha2New < alpha2LowerBound) {
               alpha2New = alpha2LowerBound;
            } else if (alpha2New > alpha2UpperBound) {
               alpha2New = alpha2UpperBound;
            }
        } else {
            float slope = lowLabel * (bHigh - bLow);
            float delta = slope * (alpha2UpperBound - alpha2LowerBound);
            if (delta > 0) {
                if (slope > 0) {
                    alpha2New = alpha2UpperBound;
                } else {
                    alpha2New = alpha2LowerBound;
                }
            } else {
                alpha2New = alpha2Old;
            }
        }

        float alpha2Diff = alpha2New - alpha2Old;
        float alpha1Diff = -sign*alpha2Diff;
        float alpha1New = alpha1Old + alpha1Diff;
        devAlphaT[iLow] = alpha2New;
        devAlphaT[iHigh] = alpha1New;
     
        *((float*)devResultT + 0) = alpha2Old;
        *((float*)devResultT + 1) = alpha1Old;
        *((float*)devResultT + 2) = bLow;
        *((float*)devResultT + 3) = bHigh;
        *((float*)devResultT + 4) = alpha2New;
        *((float*)devResultT + 5) = alpha1New;
        *((int*)devResultT + 6) = iLow;
        *((int*)devResultT + 7) = iHigh;
    }
}

void launchInitialization(float* devData,
                          int devDataPitchInFloats,
                          int nPoints, int nDimension, int kType,
                          float parameterA, float parameterB, float parameterC,
                          float* devKernelDiag, float* devAlphaT, float* devF,
                          float* devLabels, dim3 blockConfig, dim3 threadConfig) {
    switch (kType) {
    case LINEAR:
        initializeArrays<Linear><<<blockConfig, threadConfig>>>(devData,
                                                                devDataPitchInFloats,
                                                                nPoints, nDimension,
                                                                parameterA,
                                                                parameterB,
                                                                parameterC,
                                                                devKernelDiag,
                                                                devAlphaT,
                                                                devF,
                                                                devLabels);
        break;
    case POLYNOMIAL:
        initializeArrays<Polynomial><<<blockConfig, threadConfig>>>(devData,
                                                                    devDataPitchInFloats,
                                                                    nPoints, nDimension,
                                                                    parameterA,
                                                                    parameterB,
                                                                    parameterC,
                                                                    devKernelDiag,
                                                                    devAlphaT,
                                                                    devF,
                                                                    devLabels);
        break;
    case GAUSSIAN:
        initializeArrays<Gaussian><<<blockConfig, threadConfig>>>(devData,
                                                                  devDataPitchInFloats,
                                                                  nPoints, nDimension,
                                                                  parameterA,
                                                                  parameterB,
                                                                  parameterC,
                                                                  devKernelDiag,
                                                                  devAlphaT,
                                                                  devF,
                                                                  devLabels);
        break;  
    case SIGMOID:
        initializeArrays<Sigmoid><<<blockConfig, threadConfig>>>(devData,
                                                                 devDataPitchInFloats,
                                                                 nPoints, nDimension,
                                                                 parameterA,
                                                                 parameterB,
                                                                 parameterC,
                                                                 devKernelDiag,
                                                                 devAlphaT,
                                                                 devF,
                                                                 devLabels);
        break;
    }
}


void launchTakeFirstStep(void* devResultT, float* devKernelDiag,
                         float* devData, int devDataPitchInFloats,
                         float* devAlphaT, float cost, int nDimension,
                         int iLow, int iHigh, int kType, 
                         float parameterA, float parameterB, float parameterC,
                         dim3 blockConfig, dim3 threadConfig) {
    switch (kType) {
    case LINEAR:
        takeFirstStep<Linear><<<blockConfig, threadConfig>>>(devResultT,
                                                             devKernelDiag,
                                                             devData,
                                                             devDataPitchInFloats,
                                                             devAlphaT,
                                                             cost,
                                                             nDimension,
                                                             iLow,
                                                             iHigh,
                                                             parameterA,
                                                             parameterB,
                                                             parameterC);
        break;
    case POLYNOMIAL:
        takeFirstStep<Polynomial><<<blockConfig, threadConfig>>>(devResultT,
                                                                 devKernelDiag,
                                                                 devData,
                                                                 devDataPitchInFloats,
                                                                 devAlphaT,
                                                                 cost,
                                                                 nDimension,
                                                                 iLow,
                                                                 iHigh,
                                                                 parameterA,
                                                                 parameterB,
                                                                 parameterC);
        break;
    case GAUSSIAN:
        takeFirstStep<Gaussian><<<blockConfig, threadConfig>>>(devResultT,
                                                               devKernelDiag,
                                                               devData,
                                                               devDataPitchInFloats,
                                                               devAlphaT,
                                                               cost,
                                                               nDimension,
                                                               iLow,
                                                               iHigh,
                                                               parameterA,
                                                               parameterB,
                                                               parameterC);
        break;  
    case SIGMOID:
        takeFirstStep<Sigmoid><<<blockConfig, threadConfig>>>(devResultT,
                                                              devKernelDiag,
                                                              devData,
                                                              devDataPitchInFloats,
                                                              devAlphaT,
                                                              cost,
                                                              nDimension,
                                                              iLow,
                                                              iHigh,
                                                              parameterA,
                                                              parameterB,
                                                              parameterC);
        break;
    }
}

// =============== FIRSTORDER =============
int firstOrderPhaseOneSize(bool iLowCompute, bool iHighCompute, int nDimension) {
	int size = 0;
	if (iHighCompute) { size+= sizeof(float) * nDimension; }
	if (iLowCompute) { size+= sizeof(float) * nDimension; }
	return min(size,(int)sizeof(float) * 2 * 1500);
}

int firstOrderPhaseTwoSize() {
	int size = 0;
	return size;
}

void launchFirstOrder(bool iLowCompute, bool iHighCompute,
                      int kType, int nPoints, int nDimension,
                      dim3 blocksConfig, dim3 threadsConfig,
                      dim3 globalThreadsConfig, float* devData,
                      int devDataPitchInFloats, float* devTransposedData,
                      int devTransposedDataPitchInFloats, float* devLabels,
                      float epsilon, float cEpsilon, float* devAlphaT,
                      float* devF, float sAlpha1Diff, float sAlpha2Diff,
                      int iLow, int iHigh, float parameterA, float parameterB,
                      float parameterC, float* devCache, int devCachePitchInFloats,
                      int iLowCacheIndex, int iHighCacheIndex,
                      int* devLocalIndicesRL, int* devLocalIndicesRH,
                      float* devLocalFsRH, float* devLocalFsRL,
                      float* devKernelDiag, void* devResultT, float cost) {

    int phaseOneSize = firstOrderPhaseOneSize(iLowCompute, iHighCompute, nDimension);
    int phaseTwoSize = firstOrderPhaseTwoSize();
    if (iLowCompute == true) {
        if (iHighCompute == true) {
            switch (kType) {
            case LINEAR:
                firstOrderPhaseOne <true, true, Linear>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case POLYNOMIAL:
                firstOrderPhaseOne <true, true, Polynomial>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case GAUSSIAN:
                firstOrderPhaseOne <true, true, Gaussian>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case SIGMOID:
               firstOrderPhaseOne <true, true, Sigmoid>
                                  <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                  (devData, devDataPitchInFloats, devTransposedData,
                                   devTransposedDataPitchInFloats, devLabels, nPoints,
                                   nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                   sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                   parameterB, parameterC, devCache, devCachePitchInFloats,
                                   iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                   devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
               break;
            }
        } else if (iHighCompute == false) {
            switch (kType) {
            case LINEAR:
                firstOrderPhaseOne <true, false, Linear>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                  (devData, devDataPitchInFloats, devTransposedData,
                                   devTransposedDataPitchInFloats, devLabels, nPoints,
                                   nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                   sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                   parameterB, parameterC, devCache, devCachePitchInFloats,
                                   iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                   devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case POLYNOMIAL:
                firstOrderPhaseOne <true, false, Polynomial>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case GAUSSIAN:
                firstOrderPhaseOne <true, false, Gaussian>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case SIGMOID:
                firstOrderPhaseOne <true, false, Sigmoid>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            }
        }
    } else if (iLowCompute == false) {
        if (iHighCompute == true) {
            switch (kType) {
            case LINEAR:
                firstOrderPhaseOne <false, true, Linear>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case POLYNOMIAL:
                firstOrderPhaseOne <false, true, Polynomial>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case GAUSSIAN:
                firstOrderPhaseOne <false, true, Gaussian>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case SIGMOID:
                firstOrderPhaseOne <false, true, Sigmoid>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            }
        } else if (iHighCompute == false) {
            switch (kType) {
            case LINEAR:
                firstOrderPhaseOne <false, false, Linear>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case POLYNOMIAL:
                firstOrderPhaseOne <false, false, Polynomial>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case GAUSSIAN:
                firstOrderPhaseOne <false, false, Gaussian>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            case SIGMOID:
                firstOrderPhaseOne <false, false, Sigmoid>
                                   <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                   (devData, devDataPitchInFloats, devTransposedData,
                                    devTransposedDataPitchInFloats, devLabels, nPoints,
                                    nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                    sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                    parameterB, parameterC, devCache, devCachePitchInFloats,
                                    iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL,
                                    devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
                break;
            }
        }
    }
    switch (kType) {
    case LINEAR:
        firstOrderPhaseTwo<Linear>
                          <<<1, globalThreadsConfig, phaseTwoSize>>>
                          (devData, devDataPitchInFloats, devTransposedData,
                           devTransposedDataPitchInFloats, devLabels,
                           devKernelDiag, devAlphaT, devResultT, cost,
                           nDimension, parameterA, parameterB, parameterC,
                           devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL,
                           devLocalFsRH, blocksConfig.x);
        break;
    case POLYNOMIAL:
        firstOrderPhaseTwo<Polynomial>
                          <<<1, globalThreadsConfig, phaseTwoSize>>>
                          (devData, devDataPitchInFloats, devTransposedData,
                           devTransposedDataPitchInFloats, devLabels,
                           devKernelDiag, devAlphaT, devResultT, cost,
                           nDimension, parameterA, parameterB, parameterC,
                           devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL,
                           devLocalFsRH, blocksConfig.x);
        break;
    case GAUSSIAN:
        firstOrderPhaseTwo<Gaussian>
                          <<<1, globalThreadsConfig, phaseTwoSize>>>
                          (devData, devDataPitchInFloats, devTransposedData,
                           devTransposedDataPitchInFloats, devLabels,
                           devKernelDiag, devAlphaT, devResultT, cost,
                           nDimension, parameterA, parameterB, parameterC,
                           devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL,
                           devLocalFsRH, blocksConfig.x);
        break;
    case SIGMOID:
        firstOrderPhaseTwo<Sigmoid>
                          <<<1, globalThreadsConfig, phaseTwoSize>>>
                          (devData, devDataPitchInFloats, devTransposedData,
                           devTransposedDataPitchInFloats, devLabels,
                           devKernelDiag, devAlphaT, devResultT, cost,
                           nDimension, parameterA, parameterB, parameterC,
                           devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL,
                           devLocalFsRH, blocksConfig.x);
        break;
    }
} 

// ================= SECONDORDER =================
int secondOrderPhaseOneSize(bool iLowCompute, bool iHighCompute, int nDimension) {
    int size = 0;
    if (iHighCompute) { size+= sizeof(float) * nDimension; }
    if (iLowCompute) { size+= sizeof(float) * nDimension; }
    return min(size,(int)sizeof(float) * 2 * 1500);
}

int secondOrderPhaseTwoSize() {
    int size = 0;
    return size;
}

int secondOrderPhaseThreeSize(bool iHighCompute, int nDimension) {
    int size = 0;
    if (iHighCompute) { size+= sizeof(float) * nDimension; }
    return min(size,(int)sizeof(float) * 1500);
}

int secondOrderPhaseFourSize() {
    int size = 0;
    return size;
}

void launchSecondOrder(bool iLowCompute, bool iHighCompute, int kType,
                       int nPoints, int nDimension, dim3 blocksConfig,
                       dim3 threadsConfig, dim3 globalThreadsConfig,
                       float* devData, int devDataPitchInFloats,
                       float* devTransposedData,
                       int devTransposedDataPitchInFloats, float* devLabels,
                       float epsilon, float cEpsilon, float* devAlphaT,
                       float* devF, float sAlpha1Diff, float sAlpha2Diff,
                       int iLow, int iHigh, float parameterA, float parameterB,
                       float parameterC, Cache* kernelCache, float* devCache,
                       int devCachePitchInFloats, int iLowCacheIndex,
                       int iHighCacheIndex, int* devLocalIndicesRH,
                       float* devLocalFsRH, float* devLocalFsRL,
                       int* devLocalIndicesMaxObj, float* devLocalObjsMaxObj,
                       float* devKernelDiag, void* devResultT,
                       float* train_result, float cost, int iteration) {

    int phaseOneSize = secondOrderPhaseOneSize(iLowCompute, iHighCompute, nDimension);
    int phaseTwoSize = secondOrderPhaseTwoSize();
 
    if (iLowCompute == true) {
        if (iHighCompute == true) {
            switch (kType) {
            case LINEAR:
                secondOrderPhaseOne <true, true, Linear>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache, devCachePitchInFloats,
                                     iLowCacheIndex, iHighCacheIndex,
                                     devLocalIndicesRH, devLocalFsRH);
                break;
            case POLYNOMIAL:
                secondOrderPhaseOne <true, true, Polynomial>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache, devCachePitchInFloats,
                                     iLowCacheIndex, iHighCacheIndex,
                                     devLocalIndicesRH, devLocalFsRH);
                break;
            case GAUSSIAN:
                secondOrderPhaseOne <true, true, Gaussian>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache, devCachePitchInFloats,
                                     iLowCacheIndex, iHighCacheIndex,
                                     devLocalIndicesRH, devLocalFsRH);
                break;
            case SIGMOID:
                secondOrderPhaseOne <true, true, Sigmoid>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache,
                                     devCachePitchInFloats, iLowCacheIndex,
                                     iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                break;
        }
      } else if (iHighCompute == false) {
          switch (kType) {
          case LINEAR:
             secondOrderPhaseOne <true, false, Linear>
                                 <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                 (devData, devDataPitchInFloats, devTransposedData,
                                  devTransposedDataPitchInFloats, devLabels, nPoints,
                                  nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                  sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                  parameterB, parameterC, devCache,
                                  devCachePitchInFloats, iLowCacheIndex,
                                  iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
             break;
          case POLYNOMIAL:
              secondOrderPhaseOne <true, false, Polynomial>
                                  <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                  (devData, devDataPitchInFloats, devTransposedData,
                                   devTransposedDataPitchInFloats, devLabels, nPoints,
                                   nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                   sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                   parameterB, parameterC, devCache,
                                   devCachePitchInFloats, iLowCacheIndex,
                                   iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
              break;
          case GAUSSIAN:
              secondOrderPhaseOne <true, false, Gaussian>
                                  <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                  (devData, devDataPitchInFloats, devTransposedData,
                                   devTransposedDataPitchInFloats, devLabels, nPoints,
                                   nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                   sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                   parameterB, parameterC, devCache,
                                   devCachePitchInFloats, iLowCacheIndex,
                                   iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
              break;
          case SIGMOID:
              secondOrderPhaseOne <true, false, Sigmoid>
                                  <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                  (devData, devDataPitchInFloats, devTransposedData,
                                   devTransposedDataPitchInFloats, devLabels, nPoints,
                                   nDimension, epsilon, cEpsilon, devAlphaT,
                                   devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                   parameterB, parameterC, devCache,
                                   devCachePitchInFloats, iLowCacheIndex,
                                   iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
              break;
          }
        }
    } else if (iLowCompute == false) {
        if (iHighCompute == true) {
            switch (kType) {
            case LINEAR:
                secondOrderPhaseOne <false, true, Linear>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache,
                                     devCachePitchInFloats, iLowCacheIndex,
                                     iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                break;
            case POLYNOMIAL:
                secondOrderPhaseOne <false, true, Polynomial>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache,
                                     devCachePitchInFloats, iLowCacheIndex,
                                     iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                break;
            case GAUSSIAN:
                secondOrderPhaseOne <false, true, Gaussian>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache,
                                     devCachePitchInFloats, iLowCacheIndex,
                                     iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                break;
            case SIGMOID:
                secondOrderPhaseOne <false, true, Sigmoid>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache,
                                     devCachePitchInFloats, iLowCacheIndex,
                                     iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                break;
            }
        } else if (iHighCompute == false) {
            switch (kType) {
            case LINEAR:
                secondOrderPhaseOne <false, false, Linear>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache,
                                     devCachePitchInFloats, iLowCacheIndex,
                                     iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                break;
            case POLYNOMIAL:
                secondOrderPhaseOne <false, false, Polynomial>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache,
                                     devCachePitchInFloats, iLowCacheIndex,
                                     iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                break;
            case GAUSSIAN:
                secondOrderPhaseOne <false, false, Gaussian>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache,
                                     devCachePitchInFloats, iLowCacheIndex,
                                     iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                break;
            case SIGMOID:
                secondOrderPhaseOne <false, false, Sigmoid>
                                    <<<blocksConfig, threadsConfig, phaseOneSize>>>
                                    (devData, devDataPitchInFloats, devTransposedData,
                                     devTransposedDataPitchInFloats, devLabels, nPoints,
                                     nDimension, epsilon, cEpsilon, devAlphaT, devF,
                                     sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA,
                                     parameterB, parameterC, devCache,
                                     devCachePitchInFloats, iLowCacheIndex,
                                     iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                break;
            }
        }
    }

    secondOrderPhaseTwo<<<1, globalThreadsConfig, phaseTwoSize>>>
                       (devResultT, devLocalIndicesRH, devLocalFsRH, blocksConfig.x);
     

    CUDA_SAFE_CALL(cudaMemcpy((void*)train_result, devResultT, 8*sizeof(float), cudaMemcpyDeviceToHost));
     
    float bHigh = *(train_result + 3);
    iHigh = *((int*)train_result + 7);
     
    kernelCache->findData(iHigh, iHighCacheIndex, iHighCompute);
 
    int phaseThreeSize = secondOrderPhaseThreeSize(iHighCompute, nDimension);
 
    if (iHighCompute == true) {
        switch (kType) {
        case LINEAR:
            secondOrderPhaseThree<true, Linear>
                                 <<<blocksConfig, threadsConfig, phaseThreeSize>>>
                                 (devData, devDataPitchInFloats, devTransposedData,
                                  devTransposedDataPitchInFloats, devLabels,
                                  devKernelDiag, epsilon, cEpsilon, devAlphaT,
                                  devF, bHigh, iHigh, devCache, devCachePitchInFloats,
                                  iHighCacheIndex, nDimension, nPoints,
                                  parameterA, parameterB, parameterC,
                                  devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
            break;
        case POLYNOMIAL:
            secondOrderPhaseThree<true, Polynomial>
                                 <<<blocksConfig, threadsConfig, phaseThreeSize>>>
                                 (devData, devDataPitchInFloats, devTransposedData,
                                  devTransposedDataPitchInFloats, devLabels,
                                  devKernelDiag, epsilon, cEpsilon, devAlphaT,
                                  devF, bHigh, iHigh, devCache, devCachePitchInFloats,
                                  iHighCacheIndex, nDimension, nPoints,
                                  parameterA, parameterB, parameterC,
                                  devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
            break;
        case GAUSSIAN:
            secondOrderPhaseThree<true, Gaussian>
                                 <<<blocksConfig, threadsConfig, phaseThreeSize>>>
                                 (devData, devDataPitchInFloats, devTransposedData,
                                  devTransposedDataPitchInFloats, devLabels,
                                  devKernelDiag, epsilon, cEpsilon, devAlphaT,
                                  devF, bHigh, iHigh, devCache, devCachePitchInFloats,
                                  iHighCacheIndex, nDimension, nPoints,
                                  parameterA, parameterB, parameterC,
                                  devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
            break;
        case SIGMOID:
            secondOrderPhaseThree<true, Sigmoid>
                                 <<<blocksConfig, threadsConfig, phaseThreeSize>>>
                                 (devData, devDataPitchInFloats, devTransposedData,
                                  devTransposedDataPitchInFloats, devLabels,
                                  devKernelDiag, epsilon, cEpsilon, devAlphaT,
                                  devF, bHigh, iHigh, devCache, devCachePitchInFloats,
                                  iHighCacheIndex, nDimension, nPoints,
                                  parameterA, parameterB, parameterC,
                                  devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
            break;
        }
    } else {
        switch (kType) {
        case LINEAR:
            secondOrderPhaseThree<false, Linear>
                                 <<<blocksConfig, threadsConfig, phaseThreeSize>>>
                                 (devData, devDataPitchInFloats, devTransposedData,
                                  devTransposedDataPitchInFloats, devLabels,
                                  devKernelDiag, epsilon, cEpsilon, devAlphaT,
                                  devF, bHigh, iHigh, devCache, devCachePitchInFloats,
                                  iHighCacheIndex, nDimension, nPoints,
                                  parameterA, parameterB, parameterC,
                                  devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
            break;
        case POLYNOMIAL:
            secondOrderPhaseThree<false, Polynomial>
                                 <<<blocksConfig, threadsConfig, phaseThreeSize>>>
                                 (devData, devDataPitchInFloats, devTransposedData,
                                  devTransposedDataPitchInFloats, devLabels,
                                  devKernelDiag, epsilon, cEpsilon, devAlphaT,
                                  devF, bHigh, iHigh, devCache, devCachePitchInFloats,
                                  iHighCacheIndex, nDimension, nPoints,
                                  parameterA, parameterB, parameterC,
                                  devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
            break;
        case GAUSSIAN:
            secondOrderPhaseThree<false, Gaussian>
                                 <<<blocksConfig, threadsConfig, phaseThreeSize>>>
                                 (devData, devDataPitchInFloats, devTransposedData,
                                  devTransposedDataPitchInFloats, devLabels,
                                  devKernelDiag, epsilon, cEpsilon, devAlphaT,
                                  devF, bHigh, iHigh, devCache, devCachePitchInFloats,
                                  iHighCacheIndex, nDimension, nPoints,
                                  parameterA, parameterB, parameterC,
                                  devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
            break;
        case SIGMOID:
           secondOrderPhaseThree<false, Sigmoid>
                                <<<blocksConfig, threadsConfig, phaseThreeSize>>>
                                (devData, devDataPitchInFloats, devTransposedData,
                                 devTransposedDataPitchInFloats, devLabels,
                                 devKernelDiag, epsilon, cEpsilon, devAlphaT,
                                 devF, bHigh, iHigh, devCache, devCachePitchInFloats,
                                 iHighCacheIndex, nDimension, nPoints,
                                 parameterA, parameterB, parameterC,
                                 devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
           break;
        }
    }

    secondOrderPhaseFour<<<1, globalThreadsConfig, phaseTwoSize>>>
                       (devLabels, devKernelDiag, devF, devAlphaT,
                        cost, iHigh, bHigh, devResultT, devCache,
                        devCachePitchInFloats, iHighCacheIndex,
                        devLocalFsRL, devLocalIndicesMaxObj,
                        devLocalObjsMaxObj, blocksConfig.x, iteration);
} 

// ====================== CLASSIFY KERNELS ============================

/************
 * This function computes self dot products (Euclidean norm squared) for every vector in an array
 * @param devSource the vectors, in column major format
 * @param devSourcePitchInFloats the pitch of each row of the vectors (this is guaranteed to be >= sourceCount.  It might be greater due to padding, to keep each row of the source vectors aligned.
 * @param devDest a vector which will receive the self dot product
 * @param sourceCount the number of vectors
 * @param sourceLength the dimensionality of each vector
 */
__global__ void makeSelfDots(float* devSource,
                             int devSourcePitchInFloats,
                             float* devDest, int sourceCount,
                             int sourceLength) {
    float dot = 0;
    int index = BLOCKSIZE * blockIdx.x + threadIdx.x;
    
    if (index < sourceCount) {
        for (int i = 0; i < sourceLength; i++) {
        	float currentElement =
                  *(devSource + IMUL(devSourcePitchInFloats, i) + index); 
           	dot = dot + currentElement * currentElement;
        }
        *(devDest + index) = dot;
    }
}

void launchMakeSelfDots(float* devSource,
                        int devSourcePitchInFloats,
                        float* devDest,
                        int sourceCount,
                        int sourceLength,
                        dim3 num_blocks,
                        dim3 num_threads) { 

    makeSelfDots<<<num_blocks, num_threads>>>(devSource,
                                              devSourcePitchInFloats, devDest,
                                              sourceCount, sourceLength);
}

/**
 * This function constructs a matrix devDots, where devDots_(i,j) = ||data_i||^2 + ||SV_j||^2
 * @param devDots the output array
 * @param devDotsPitchInFloats the pitch of each row of devDots.  Guaranteed to be >= nSV
 * @param devSVDots a vector containing ||SV_j||^2 for all j in [0, nSV - 1]
 * @param devDataDots a vector containing ||data_i||^2 for all i in [0, nPoints - 1]
 * @param nSV the number of Support Vectors in the classifier
 */
__global__ void makeDots(float* devDots, int devDotsPitchInFloats, float* devSVDots, float* devDataDots, int nSV, int nPoints) {
    __shared__ float localSVDots[BLOCKSIZE];
    __shared__ float localDataDots[BLOCKSIZE];
    int svIndex = IMUL(BLOCKSIZE, blockIdx.x) + threadIdx.x;
    
    if (svIndex < nSV) {
   	     localSVDots[threadIdx.x] = *(devSVDots + svIndex);
    }
    
    int dataIndex = BLOCKSIZE * blockIdx.y + threadIdx.x;
    if (dataIndex < nPoints) {
         localDataDots[threadIdx.x] = *(devDataDots + dataIndex);
    }
    
    __syncthreads();
    
    dataIndex = BLOCKSIZE * blockIdx.y;
    for(int i = 0; i < BLOCKSIZE; i++, dataIndex++) {
         if ((svIndex < nSV) && (dataIndex < nPoints)) {
              *(devDots + IMUL(devDotsPitchInFloats, dataIndex) + svIndex) =
                          localSVDots[threadIdx.x] + localDataDots[i];
         }
    }
}

void launchMakeDots(float* devDots,
                    int devDotsPitchInFloats,
                    float* devSVDots,
                    float* devDataDots,
                    int nSV,
                    int nPoints,
                    dim3 num_blocks,
                    dim3 num_threads) {

    makeDots<<<num_blocks, num_threads>>>(devDots, devDotsPitchInFloats,
                                          devSVDots, devDataDots, nSV, nPoints);
}

__device__ void computeKernels(float* devNorms, int devNormsPitchInFloats,
                               float* devAlphaC, int nPoints, int nSV,
                               int kernelType, float coef0, int degree,
                               float* localValue, int svIndex) {
    if (svIndex < nSV) {
   	     float alpha = devAlphaC[svIndex];
         float norm = devNorms[IMUL(devNormsPitchInFloats, blockIdx.y) + svIndex];
   	     if(kernelType == GAUSSIAN)
   	     {
             localValue[threadIdx.x] = alpha * exp(norm);
   	     }
         else if(kernelType == LINEAR)
         {
             localValue[threadIdx.x] = alpha * norm;
         }
         else if(kernelType == POLYNOMIAL)
         {
              localValue[threadIdx.x] = alpha * pow(norm + coef0, degree);
         }
         else if(kernelType == SIGMOID)
         {
              localValue[threadIdx.x] = alpha * tanh(norm + coef0);
         }
    }
}

/**
 * This function completes the kernel evaluations and begins the reductions to form the classification result.
 * @param devNorms this contains partially completed kernel evaluations.  For most kernels, devNorms_(i, j) = data_i (dot) sv_j.  For the RBF kernel, devNorms_(i, j) = -gamma*(||data_i||^2 + ||sv_j||^2 - 2* data_i (dot) sv_j)
 * @param devNormsPitchInFloats contains the pitch of the partially completed kernel evaluations.  It will always be >= nSV.
 * @param devAlphaC this is the alpha vector for the SVM classifier
 * @param nPoints the number of data points
 * @param nSV the number of support vectors
 * @param kernelType the type of kernel
 * @param coef0 a coefficient used in the polynomial & sigmoid kernels
 * @param degree the degree used in the polynomial kernel
 * @param devLocalValue the local classification results
 * @param reduceOffset computed to begin the reduction properly
 */
__global__ void computeKernelsReduce(float* devNorms,
                                     int devNormsPitchInFloats,
                                     float* devAlphaC, int nPoints,
                                     int nSV, int kernelType,
                                     float coef0, int degree,
                                     float* devLocalValue,
                                     int reduceOffset) {
	/*Dynamic shared memory setup*/
    extern __shared__ float localValue[];
    int svIndex = blockDim.x * blockIdx.x + threadIdx.x;
  
    computeKernels(devNorms, devNormsPitchInFloats,
                   devAlphaC, nPoints, nSV, kernelType,
                   coef0, degree, localValue, svIndex);
    __syncthreads();
	
   /*reduction*/
    for(int offset = reduceOffset; offset >= 1; offset = offset >> 1) {
        if ((threadIdx.x < offset) && (svIndex + offset < nSV)) {
            int compOffset = threadIdx.x + offset;
            localValue[threadIdx.x] = localValue[threadIdx.x] + localValue[compOffset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
       devLocalValue[blockIdx.x + gridDim.x*blockIdx.y] = localValue[0];
    }
}

void launchComputeKernelsReduce(float* devNorms,
                                int devNormsPitchInFloats,
                                float* devAlphaC,
                                int nPoints, int nSV,
                                int kernelType, float coef0,
                                int degree, float* devLocalValue,
                                int reduceOffset,
                                dim3 num_blocks,
                                dim3 num_threads,
                                int shared_size) {

    computeKernelsReduce<<<num_blocks, num_threads, shared_size>>>(devNorms, 
                                                                   devNormsPitchInFloats,
                                                                   devAlphaC, 
                                                                   nPoints, nSV, 
                                                                   kernelType, coef0,
                                                                   degree, devLocalValue,
                                                                   reduceOffset);
}

/*Second stage reduce and cleanup function*/ 
__global__ void doClassification(float* devResultC, float b,
                                 float* devLocalValue,
                                 int reduceOffset, int nPoints) {
	
    extern __shared__ float localValue[];
	
    localValue[threadIdx.x] = devLocalValue[blockDim.x*blockIdx.y + threadIdx.x];
    __syncthreads();
    for(int offset = reduceOffset; offset >= 1; offset = offset >> 1) {
        if (threadIdx.x < offset) {
            int compOffset = threadIdx.x + offset;
            if (compOffset < blockDim.x) {
                localValue[threadIdx.x] =
                    localValue[threadIdx.x] + localValue[compOffset];
            }
        }
        __syncthreads();
    }

	float sumResult = localValue[0];
	if (threadIdx.x == 0) {
		sumResult += b;
		devResultC[blockIdx.y] = sumResult;
	}
}

void launchDoClassification(float* devResultC,
                            float b,
                            float* devLocalValue,
                            int reduceOffset,
                            int nPoints,
                            dim3 num_blocks,
                            dim3 num_threads,
                            int shared_size) {

    doClassification<<<num_blocks, num_threads, shared_size>>>(devResultC, b,
                                                               devLocalValue,
                                                               reduceOffset,
                                                               nPoints);
}
