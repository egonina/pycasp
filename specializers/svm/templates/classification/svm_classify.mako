// ======================== SVM CLASSIFY ================
#define MAX_PITCH 262144
#define MAX_POINTS (MAX_PITCH/sizeof(float) - 2)

#define min(a, b) (a < b) ? a : b

// TODO: change parameter list
void classify(int nData,
              int nDimension,
              int nSV,
              int kernel_type, float gamma,
              float coef0, float degree,
              float b)
{
    int total_nPoints = nData;
    int nPoints;	
    
    printf("......Testing SVM......\n");
    
    if(kernel_type == 0) {
        gamma = 1.0;
    }
    
    int nBlocksSV = intDivideRoundUp(nSV, ${num_threads});
    
    cublasStatus status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
       	printf("CUBLAS initialization error\n");
       	exit(0);
    }
    
    float* devLocalValue;
    float* devSVDots;
    CUDA_SAFE_CALL(cudaMalloc((void**)&devSVDots, sizeof(float) * nSV));
    
    size_t free_memory,total_memory;
    cuMemGetInfo(&free_memory,&total_memory);
    int free_memory_floats = (long int)free_memory/sizeof(float);
    
    free_memory_floats = (int)(0.9 * free_memory_floats); 
    
    nPoints = ((free_memory_floats - devSVPitchInFloats * nDimension - nSV - nSV) /
              (nDimension + 1 + devSVPitchInFloats + 1 + nBlocksSV));
    nPoints = (nPoints >> 7) << 7;   //for pitch limitations assigning to be a multiple of 128
    
    nPoints = min(nPoints, total_nPoints); //for few points
    nPoints = min(nPoints, (int)MAX_POINTS); //for too many points	
    
    //printf("Max points that can reside in GPU memory per call = %d\n\n", nPoints);
    
    dim3 mapGrid(intDivideRoundUp(nSV, ${num_threads}), nPoints);
    dim3 mapBlock(${num_threads});
    
    dim3 reduceGrid(1, nPoints);
    dim3 reduceBlock(mapGrid.x, 1);
    
    int devDataPitchInFloats = devDataPitch/sizeof(float);
    float* devDataDots;
    CUDA_SAFE_CALL(cudaMalloc((void**)&devDataDots,
                              sizeof(float) * nPoints));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalValue,
                              sizeof(float) * mapGrid.x * mapGrid.y));
    
    float* devDots;
    size_t devDotsPitch;
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&devDots, &devDotsPitch,
                                   nSV * sizeof(float), nPoints));
    
    dim3 threadsLinear(${num_threads});
    if(kernel_type == 1) {
    	dim3 blocksSVLinear(intDivideRoundUp(nSV, ${num_threads}));
    	launchMakeSelfDots(devSV, devSVPitchInFloats,
                           devSVDots, nSV, nDimension,
                           blocksSVLinear, threadsLinear);
    }
    
    int iteration = 1;
    
    for(int dataoffset = 0; dataoffset < total_nPoints; dataoffset += nPoints) {
        // code for copying data
        if(dataoffset+nPoints > total_nPoints) {
            nPoints = total_nPoints-dataoffset;
            mapGrid=dim3(intDivideRoundUp(nSV, ${num_threads}), nPoints);
            mapBlock=dim3(${num_threads});
            
            reduceGrid=dim3(1, nPoints);
            reduceBlock=dim3(mapGrid.x, 1);
            
            CUDA_SAFE_CALL(cudaFree(devLocalValue));
            CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalValue,
                                      sizeof(float) * mapGrid.x * mapGrid.y));
            
            //resize & copy devdata, devdots,
            CUDA_SAFE_CALL(cudaFree(devData));
            CUDA_SAFE_CALL(cudaMallocPitch((void**)&devData, &devDataPitch,
                                           nPoints * sizeof(float), nDimension));
            devDataPitchInFloats = devDataPitch/sizeof(float);
        }
        
        if(total_nPoints*sizeof(float) < MAX_PITCH) {
            CUDA_SAFE_CALL(cudaMemcpy2D(devData, devDataPitch,
                                        data+dataoffset, total_nPoints * sizeof(float),
                                        nPoints * sizeof(float), nDimension,
                                        cudaMemcpyHostToDevice));
        } else {
            for(int nd = 0; nd < nDimension; nd++) {
                 CUDA_SAFE_CALL(cudaMemcpy(devData + nd * devDataPitchInFloats,
                                           data + nd * total_nPoints + dataoffset,
                                           nPoints * sizeof(float),
                                           cudaMemcpyHostToDevice));	
            }
        }
        
        dim3 blocksDataLinear(intDivideRoundUp(nPoints, ${num_threads}));
        dim3 threadsDots(${num_threads}, 1);
        dim3 blocksDots(intDivideRoundUp(nSV, ${num_threads}),
                        intDivideRoundUp(nPoints, ${num_threads}));
        int devDotsPitchInFloats = ((int)devDotsPitch) / sizeof(float);
        
        if(kernel_type == 1) {
            launchMakeSelfDots(devData, devDataPitchInFloats,
                               devDataDots, nPoints, nDimension,
                               blocksDataLinear, threadsLinear);
            
            CUDA_SAFE_CALL(cudaMemset(devDots, 0,
                           sizeof(float) * devDotsPitchInFloats * nPoints));
            
            launchMakeDots(devDots, devDotsPitchInFloats,
                           devSVDots, devDataDots, nSV,
                           nPoints, blocksDots, threadsDots);
            
            cudaThreadSynchronize(); //unnecessary..onyl for timing..
        }
        
        float sgemmAlpha, sgemmBeta;
        if(kernel_type == 1) {
        	sgemmAlpha = 2 * gamma;
        	sgemmBeta = -gamma;
        } else {
        	sgemmAlpha = gamma;
        	sgemmBeta = 0.0f;
        }

        cublasSgemm('n', 't', nSV, nPoints,
                    nDimension, sgemmAlpha, devSV,
                    devSVPitchInFloats, devData,
                    devDataPitchInFloats, sgemmBeta,
                    devDots, devDotsPitchInFloats);
        
        cudaThreadSynchronize();
        
        int reduceOffset = (int)pow(2, ceil(log2((float)${num_threads})) - 1);
        int sharedSize = sizeof(float)*(${num_threads});
        
        if(kernel_type == 1) {
            launchComputeKernelsReduce(devDots, devDotsPitchInFloats,
                                       devAlphaC, nPoints, nSV, GAUSSIAN, 0,1,
                                       devLocalValue,
                                       1<<int(ceil(log2((float)${num_threads})) - 1),
                                       mapGrid, mapBlock, sharedSize);
        } else if(kernel_type == 2) {
            launchComputeKernelsReduce(devDots, devDotsPitchInFloats,
                                       devAlphaC, nPoints, nSV, POLYNOMIAL,
                                       coef0, degree, devLocalValue,
                                       1<<int(ceil(log2((float)${num_threads})) - 1),
                                       mapGrid, mapBlock, sharedSize);
        } else if(kernel_type == 0) {
            launchComputeKernelsReduce(devDots, devDotsPitchInFloats,
                                       devAlphaC, nPoints, nSV, LINEAR, 0,1,
                                       devLocalValue,
                                       1<<int(ceil(log2((float)${num_threads})) - 1),
                                       mapGrid, mapBlock, sharedSize);
        } else if(kernel_type == 3) {
           	launchComputeKernelsReduce(devDots, devDotsPitchInFloats,
                                       devAlphaC, nPoints, nSV, SIGMOID, coef0, 1,
                                       devLocalValue,
                                       1<<int(ceil(log2((float)${num_threads})) - 1),
                                       mapGrid, mapBlock, sharedSize);
        }
        
        
        reduceOffset = (int)pow(2, ceil(log2((float)mapGrid.x)) - 1);
        sharedSize = sizeof(float) * mapGrid.x;
 
        launchDoClassification(devResultC, b, devLocalValue,
                               reduceOffset, mapGrid.x, reduceGrid,
                               reduceBlock, sharedSize);
        
        cudaThreadSynchronize(); //unnecessary..only for timing..
        cudaMemcpy(classify_result + dataoffset,
                   devResultC, nPoints * sizeof(float),
                   cudaMemcpyDeviceToHost);
        iteration++;
    }
    
    float class1Label = 1.0;
    float class2Label = -1.0;
    
    int confusionMatrix[] = {0, 0, 0, 0};
    for (int i = 0; i < total_nPoints; i++) {
   	    if ((labels[i] == class2Label) && (classify_result[i] < 0)) {
   	    	confusionMatrix[0]++;
   	    } else if ((labels[i] == class2Label) && (classify_result[i] >= 0)) {
   	    	confusionMatrix[1]++;
   	    } else if ((labels[i] == class1Label) && (classify_result[i] < 0)) {
   	    	confusionMatrix[2]++;
   	    } else if ((labels[i] == class1Label) && (classify_result[i] >= 0)) {
   	    	confusionMatrix[3]++;
   	    }
    }
    printf("INFO: Accuracy: %f% (%d / %d) \n",
           (float)(confusionMatrix[0] + confusionMatrix[3]) * 100.0 / ((float)total_nPoints),
           confusionMatrix[0] + confusionMatrix[3], total_nPoints);
    
    CUDA_SAFE_CALL(cudaFree(devLocalValue));
    CUDA_SAFE_CALL(cudaFree(devDots));
    CUDA_SAFE_CALL(cudaFree(devSVDots));
    CUDA_SAFE_CALL(cudaFree(devDataDots));
}
