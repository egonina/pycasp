import numpy as np

# gen training data
Ds = [19, 39]
Ms = [16, 32, 64, 128, 256, 512, 1024]

NF = 20

print "gen training data"
for D in Ds:
    for M in Ms:
        N = D*M
        print N
        feats = np.random.randn(NF, N)
        labels_min_1 = np.ones(NF/2)*-1
        labels_1 = np.ones(NF/2)
        labels = np.concatenate((labels_1, labels_min_1))

        out_file = open("svm_feats/training/svm_training_feats_nf_"+str(NF)+"_D_"+str(D)+"_M_"+str(M)+".svm", "w")

        row_count = 0
        for i in feats:
            out_file.write(str(labels[row_count])+" ")
            col_count = 1
            size = i.shape[0]
            for j in i[:size-1]:
                out_file.write(str(col_count)+":"+str(j)+" ")
                col_count += 1
            out_file.write(str(col_count)+":"+str(i[size-1])+"\n")
            row_count += 1

        out_file.close()

# gen testing data
print "gen testing data"
for D in Ds:
    for M in Ms:
        N = D*M
        print N
        feats = np.random.randn(N)
        labels = np.ones(1)

        out_file = open("svm_feats/testing/svm_testing_feats_D_"+str(D)+"_M_"+str(M)+".svm", "w")

        row_count = 0
        out_file.write(str(labels[row_count])+" ")
        for i in feats[:N-1]:
            out_file.write(str(col_count)+":"+str(j)+" ")
            col_count += 1
        out_file.write(str(col_count)+":"+str(feats[N-1])+"\n")

        out_file.close()
