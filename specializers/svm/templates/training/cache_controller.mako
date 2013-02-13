// From Cache.cpp
Cache::DirectoryEntry::DirectoryEntry() {
    status = NEVER;
}

Cache::Cache(int nPointsIn, int cacheSizeIn) {
    directory.resize(nPointsIn);
    nPoints = nPointsIn;
    cacheSize = cacheSizeIn;
    occupancy = 0;
    hits = 0;
    compulsoryMisses = 0;
    capacityMisses = 0;
}

Cache::~Cache() {}

void Cache::search(const int index, int &offset, bool &compute) {
    DirectoryEntry currentEntry = directory[index];
    if (currentEntry.status == DirectoryEntry::INCACHE) {
        offset = currentEntry.location;
        compute = false;
        return;
    }
    compute = true;
    return;
}

void Cache::findData(const int index, int &offset, bool &compute) {
    std::vector<DirectoryEntry>::iterator iCurrentEntry = directory.begin() + index;
    if (iCurrentEntry->status == DirectoryEntry::INCACHE) {
        hits++;
        if (iCurrentEntry->lruListEntry == lruList.begin()) {
            offset = iCurrentEntry->location;
            compute = false;
            return;
        }
        lruList.erase(iCurrentEntry->lruListEntry);
        lruList.push_front(index);
        iCurrentEntry->lruListEntry = lruList.begin();
        offset = iCurrentEntry->location;
        compute = false;
        return;
    }
    
    //Cache Miss
    if (occupancy < cacheSize) {
        //Cache has empty space
        compulsoryMisses++;
        iCurrentEntry->location = occupancy;
        iCurrentEntry->status = DirectoryEntry::INCACHE;
        lruList.push_front(index);
        iCurrentEntry->lruListEntry = lruList.begin();
        occupancy++;
        offset = iCurrentEntry->location;
        compute = true;
        return;
    }
    
    //Cache is full
    if (iCurrentEntry->status == DirectoryEntry::NEVER) {
        compulsoryMisses++;
    } else {
        capacityMisses++;
    }
    
    int expiredPoint = lruList.back();
    lruList.pop_back();
    
    directory[expiredPoint].status = DirectoryEntry::EVICTED;
    int expiredLine = directory[expiredPoint].location;
    iCurrentEntry->status = DirectoryEntry::INCACHE;
    iCurrentEntry->location = expiredLine;
    lruList.push_front(index);
    iCurrentEntry->lruListEntry = lruList.begin();
    
    offset = iCurrentEntry->location;
    compute = true;
    return;
}

void Cache::printStatistics() {
	int accesses = hits + compulsoryMisses + capacityMisses;
	printf("%d accesses, %d hits, %d compulsory misses, %d capacity misses\n",
           accesses, hits, compulsoryMisses, capacityMisses);
	return;
}

void Cache::printCache() {
    int accesses = hits + compulsoryMisses + capacityMisses;
    float hitPercent = (float)hits*100.0/float(accesses);
    float compulsoryPercent = (float)compulsoryMisses*100.0/float(accesses);
    float capacityPercent = (float)capacityMisses*100.0/float(accesses);
    
    printf("Cache hits: %f compulsory misses: %f capacity misses %f\n",
           hitPercent, compulsoryPercent, capacityPercent);
    for(int i = 0; i < nPoints; i++) {
        if (directory[i].status == DirectoryEntry::INCACHE) {
            printf("Row %d: present @ cache line %d\n", i, directory[i].location);
        } else {
            printf("Row %d: not present\n", i);
        }
    }
    printf("----\n");
    std::list<int>::iterator i = lruList.begin();
    for(;i != lruList.end(); i++) {
        printf("Offset: %d\n", *i);
    }
}

// From Controller.cpp
Controller::Controller(float initialGap, int currentMethodIn,
                       int samplingIntervalIn, int problemSize) {
    progress.push_back(initialGap);
    currentMethod = currentMethodIn;
    if (currentMethod == ADAPTIVE) {
        adaptive = true;
        currentMethod = SECONDORDER;
    } else {
        adaptive = false;
    }
    samplingInterval = samplingIntervalIn;
    inspectionPeriod = problemSize/(10*samplingInterval);
    
    timeSinceInspection = inspectionPeriod - 2;
    beginningOfEpoch = 0;
    rates.push_back(0);
    rates.push_back(0);
    currentInspectionPhase = 0;
    //printf("Controller: currentMethod: %i (%s), inspectionPeriod: %i\n",
    //      currentMethod, adaptive?"dynamic":"static", inspectionPeriod);
}

void Controller::addIteration(float gap) {
    progress.push_back(gap);
    method.push_back(currentMethod);
}

float Controller::findRate(struct timeval* start, struct timeval* finish,
                           int beginning, int end) {
    float time = ((float)(finish->tv_sec - start->tv_sec)) * 1000000 +
                 ((float)(finish->tv_usec - start->tv_usec));
    int length = end - beginning;
    int filterLength = length / 2;
    float phase1Gap = filter(beginning, beginning + filterLength);
    float phase2Gap = filter(beginning + filterLength, end);
    float percentageChange = (phase2Gap - phase1Gap)/phase1Gap;
    float percentRate = percentageChange / time;
    //printf("%f\n", percentRate);
    return percentRate;
}

int Controller::getMethod() {
    if (!adaptive) {
        if (currentMethod == RANDOM) {
            if ((rand() & 0x1) > 0) {
                return SECONDORDER;
            } else {
                return FIRSTORDER;
            }
        }
        return currentMethod;
    }

    if (timeSinceInspection >= inspectionPeriod) {
        int currentIteration = progress.size();
        gettimeofday(&start, 0);
        currentInspectionPhase = 1;
        timeSinceInspection = 0;
        beginningOfEpoch = currentIteration;
    } else if (currentInspectionPhase == 1) {
        int currentIteration = progress.size();

        middleOfEpoch = currentIteration;
        gettimeofday(&mid, 0);
        rates[currentMethod] = findRate(&start, &mid, beginningOfEpoch, middleOfEpoch);
        currentInspectionPhase++;

        if (currentMethod == FIRSTORDER) {
            currentMethod = SECONDORDER;
        } else {
            currentMethod = FIRSTORDER;
        }
    } else if (currentInspectionPhase == 2) {
        int currentIteration = progress.size();
            
        gettimeofday(&finish, 0);
        rates[currentMethod] = findRate(&mid, &finish, middleOfEpoch, currentIteration);
        timeSinceInspection = 0;
        currentInspectionPhase = 0;
        
        if (fabs(rates[1]) > fabs(rates[0])) {
            currentMethod = SECONDORDER;
        } else {
            currentMethod = FIRSTORDER;
        }
        //printf("Rate 0: %f, Rate 1: %f, choose method: %i\n", rates[0], rates[1], currentMethod);
    } else {
        timeSinceInspection++;
    }
    return currentMethod;
}

float Controller::filter(int begin, int end) {
    float accumulant = 0;
    for (int i = begin; i < end; i++) {
        accumulant += progress[i];
    }
    accumulant = accumulant / ((float)(end - begin));
    return accumulant;
}

void Controller::print() {
    FILE* outputFilePointer = fopen("gap.dat", "w");
    if (outputFilePointer == NULL) {
        printf("Can't write %s\n", "gap.dat");
        exit(1);
    }
    for(vector<float>::iterator i = progress.begin(); i != progress.end(); i++) {
        fprintf(outputFilePointer, "%f ", *i);
    }
    fprintf(outputFilePointer, "\n");
    fclose(outputFilePointer);

    outputFilePointer = fopen("method.dat", "w");
    if (outputFilePointer == NULL) {
        printf("Can't write %s\n", "method.dat");
        exit(1);
    }
    for(vector<int>::iterator i = method.begin(); i != method.end(); i++) {
        fprintf(outputFilePointer, "%d ", *i);
    }
    fprintf(outputFilePointer, "\n");
    fclose(outputFilePointer);
}
