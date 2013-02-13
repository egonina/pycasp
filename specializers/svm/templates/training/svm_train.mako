// ======================== SVM TRAIN ================
PyObject* train(int nPoints, int nDimension,
                int kernel_type, float gamma,
                float coef0, float degree,
                float cost, int heuristicMethod,
                float epsilon, float tolerance) {
    
    float cEpsilon = cost - epsilon;
    Controller progress(2.0, heuristicMethod, 64, nPoints);

    printf("......Training SVM......\n");
    // Determine kernel type and parameters
    int kType = GAUSSIAN;
    float parameterA;
    float parameterB;
    float parameterC;
    if (kernel_type == 1) {
        parameterA = -gamma;
        kType = GAUSSIAN;
        printf("Gaussian kernel: gamma = %f\n", -parameterA);
    } else if (kernel_type == 2) {
        parameterA = gamma;
        parameterB = coef0;
        parameterC = degree;
        kType = POLYNOMIAL;
        printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterA, parameterB, parameterC);
    } else if (kernel_type == 0) {
        kType = LINEAR;
        printf("Linear kernel\n");
    } else if (kernel_type == 3) {
        kType = SIGMOID;
        parameterA = gamma;
        parameterB = coef0;
        printf("Sigmoid kernel: a = %f, r = %f\n", parameterA, parameterB);
        if ((parameterA <= 0) || (parameterB < 0)) {
            printf("Invalid Parameters\n");
            exit(1);
        }
    }
    
    // Determine the number of thread blocks based on the number of training points
    int blockWidth = intDivideRoundUp(nPoints, ${num_threads});
    
    // Allocate kernel diagonal elements (?) on the GPU
    float* devKernelDiag;
    CUDA_SAFE_CALL(cudaMalloc((void**)&devKernelDiag, nPoints * sizeof(float)));
    
    // Allocate the error array on the GPU
    float* devF;
    CUDA_SAFE_CALL(cudaMalloc((void**)&devF, nPoints*sizeof(float)));
    
    // Allocate helper data structures on the GPU
    float* devLocalFsRL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalFsRL, blockWidth * sizeof(float)));
    float* devLocalFsRH;
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalFsRH, blockWidth * sizeof(float))); 
    int* devLocalIndicesRL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalIndicesRL, blockWidth * sizeof(int)));
    int* devLocalIndicesRH;
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalIndicesRH, blockWidth * sizeof(int)));
    
    float* devLocalObjsMaxObj;
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalObjsMaxObj, blockWidth * sizeof(float)));
    int* devLocalIndicesMaxObj;
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalIndicesMaxObj, blockWidth * sizeof(int)));
    void* temp;
    size_t rowPitch;
    CUDA_SAFE_CALL(cudaMallocPitch(&temp, &rowPitch, nPoints * sizeof(float), 2));
    CUDA_SAFE_CALL(cudaFree(temp));
    
    // Determine the size of the cache
    size_t remainingMemory;
    size_t totalMemory;
    cuMemGetInfo(&remainingMemory, &totalMemory);
    
    int sizeOfCache = remainingMemory / ((int)rowPitch);
    //If I try to grab all the memory available, it'll fail
    sizeOfCache = (int)((float)sizeOfCache * 0.95);
    if (nPoints < sizeOfCache) { sizeOfCache = nPoints; }
    
    // Allocate cache on the GPU
    float* devCache;
    size_t cachePitch;
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&devCache, &cachePitch,
                                   nPoints * sizeof(float), sizeOfCache));
    Cache kernelCache(nPoints, sizeOfCache);
    int devCachePitchInFloats = (int)cachePitch / (sizeof(float));
    
    cudaError_t err = cudaGetLastError();
    if(err) printf("Error: %s\n", cudaGetErrorString(err));
    
    // Set number of blocks and number of threads per block
    dim3 threadsLinear(${num_threads});
    dim3 blocksLinear(blockWidth);
    
    int devDataPitchInFloats = ((int)devDataPitch) >> 2;
    int devTransposedDataPitchInFloats = ((int)devTransposedDataPitch) >> 2;
      
    // Initialize
    launchInitialization(devData, devDataPitchInFloats,
                         nPoints, nDimension, kType, parameterA,
                         parameterB, parameterC, devKernelDiag,
                         devAlphaT, devF, devLabels,
                         blocksLinear, threadsLinear);

    err = cudaGetLastError();
    if(err) printf("Error: %s\n", cudaGetErrorString(err));
    printf(".....Initialization complete.....\n");
    
    //Choose initial points
    float bLow = 1;
    float bHigh = -1;
    int iteration = 0;
    int iLow = -1;
    int iHigh = -1;
    for (int i = 0; i < nPoints; i++) {
        if (labels[i] < 0) {
            if (iLow == -1) {
                iLow = i;
                if (iHigh > -1) {
                    i = nPoints; //Terminate
                }
            }
    	} else {
            if (iHigh == -1) {
                iHigh = i;
                if (iLow > -1) {
                    i = nPoints; //Terminate
                }
            }
        }
    }
    
    dim3 singletonThreads(1);
    dim3 singletonBlocks(1);

    launchTakeFirstStep(devResultT, devKernelDiag,
                        devData, devDataPitchInFloats,
                        devAlphaT, cost, nDimension,
                        iLow, iHigh, kType, parameterA,
                        parameterB, parameterC,
                        singletonBlocks, singletonThreads);

    CUDA_SAFE_CALL(cudaMemcpy((void*)train_result, devResultT,
                   8 * sizeof(float), cudaMemcpyDeviceToHost));
    
    float alpha2Old = *(train_result + 0);
    float alpha1Old = *(train_result + 1);
    bLow = *(train_result + 2);
    bHigh = *(train_result + 3);
    float alpha2New = *(train_result + 6);
    float alpha1New = *(train_result + 7);
    
    float alpha1Diff = alpha1New - alpha1Old;
    float alpha2Diff = alpha2New - alpha2Old;
    
    int iLowCacheIndex;
    int iHighCacheIndex;
    bool iLowCompute;
    bool iHighCompute; 
    
    dim3 reduceThreads(${num_threads});
      
    for (iteration = 1; true; iteration++) {
       if (bLow <= bHigh + 2 * tolerance) {
           break; //Convergence!!
       }
       
       if ((iteration & 0x7ff) == 0) {
           printf("iteration: %d; gap: %f\n",iteration, bLow - bHigh);
       }
       
       if ((iteration & 0x7f) == 0) {
           heuristicMethod = progress.getMethod();
       }
       
       kernelCache.findData(iHigh, iHighCacheIndex, iHighCompute);
       kernelCache.findData(iLow, iLowCacheIndex, iLowCompute);
       
       if (heuristicMethod == FIRSTORDER) {
           launchFirstOrder(iLowCompute, iHighCompute,
                            kType, nPoints, nDimension,
                            blocksLinear, threadsLinear,
                            reduceThreads, devData,
                            devDataPitchInFloats, devTransposedData,
                            devTransposedDataPitchInFloats, devLabels,
                            epsilon, cEpsilon, devAlphaT, devF,
                            alpha1Diff * labels[iHigh],
                            alpha2Diff * labels[iLow], iLow, iHigh,
                            parameterA, parameterB, parameterC,
                            devCache, devCachePitchInFloats,
                            iLowCacheIndex, iHighCacheIndex,
                            devLocalIndicesRL, devLocalIndicesRH,
                            devLocalFsRH, devLocalFsRL,
                            devKernelDiag, devResultT, cost);
       } else {
           launchSecondOrder(iLowCompute, iHighCompute,
                             kType, nPoints, nDimension,
                             blocksLinear, threadsLinear,
                             reduceThreads, devData,
                             devDataPitchInFloats, devTransposedData,
                             devTransposedDataPitchInFloats, devLabels,
                             epsilon, cEpsilon, devAlphaT, devF,
                             alpha1Diff * labels[iHigh],
                             alpha2Diff * labels[iLow], iLow, iHigh,
                             parameterA, parameterB, parameterC,
                             &kernelCache, devCache, devCachePitchInFloats,
                             iLowCacheIndex, iHighCacheIndex,
                             devLocalIndicesRH, devLocalFsRH, devLocalFsRL,
                             devLocalIndicesMaxObj, devLocalObjsMaxObj,
                             devKernelDiag, devResultT,
                             train_result, cost, iteration);
       }

       CUDA_SAFE_CALL(cudaMemcpy((void*)train_result, devResultT,
                                 8 * sizeof(float), cudaMemcpyDeviceToHost));
       alpha2Old = *(train_result + 0);
       alpha1Old = *(train_result + 1);
       bLow = *(train_result + 2);
       bHigh = *(train_result + 3);
       iLow = *((int*)train_result + 6);
       iHigh = *((int*)train_result + 7);
       alpha2New = *(train_result + 4);
       alpha1New = *(train_result + 5);
       alpha1Diff = alpha1New - alpha1Old;
       alpha2Diff = alpha2New - alpha2Old;
       progress.addIteration(bLow-bHigh);
       CUT_CHECK_ERROR("SMO Iteration Failed");
    }
      
    printf("INFO: %d iterations\n", iteration);
    printf("INFO: bLow: %f, bHigh: %f\n", bLow, bHigh);
    //kernelCache.printStatistics();
    CUDA_SAFE_CALL(cudaMemcpy((void*)alphaT, devAlphaT,
                              nPoints * sizeof(float), cudaMemcpyDeviceToHost)); 
    
    // Deallocate all locally allocated GPU memory
    cudaFree(devF);
    cudaFree(devCache);
    cudaFree(devLocalIndicesRL);
    cudaFree(devLocalIndicesRH);
    cudaFree(devLocalFsRH);
    cudaFree(devLocalFsRL);
    cudaFree(devKernelDiag);
    cudaFree(devLocalIndicesMaxObj);
    cudaFree(devLocalObjsMaxObj);

    float* sv;
    float* out_a;
    int total_sv = storeModel(kernel_type, gamma, coef0,
                              degree, alphaT, labels,
                              data, nPoints,
                              nDimension, epsilon,
                              &sv, &out_a); 

    npy_intp npLen_sv[1];
    npy_intp npLen_a[1];
    npLen_sv[0] = {total_sv * nDimension};
    npLen_a[0] = {total_sv};
    PyObject* support_vectors = PyArray_SimpleNewFromData(1, npLen_sv, PyArray_FLOAT, sv);
    PyObject* out_alphas = PyArray_SimpleNewFromData(1, npLen_a, PyArray_FLOAT, out_a);
    PyObject* rho = (PyObject*)Py_BuildValue("f", -(((bHigh + bLow) / 2)));
    PyObject* nsv = (PyObject*)Py_BuildValue("i", total_sv);

    PyTupleObject *ret = (PyTupleObject*)PyTuple_New(4);
    PyTuple_SET_ITEM(ret, 0, support_vectors);
    PyTuple_SET_ITEM(ret, 1, out_alphas);
    PyTuple_SET_ITEM(ret, 2, rho);
    PyTuple_SET_ITEM(ret, 3, nsv);

    return (PyObject*)ret;
}
