import numpy as np

D = 19
Ns = [50000]

for N in Ns:
    feats = np.random.randn(N, D)
    labels_min_1 = np.ones(N/2)*-1
    labels_1 = np.ones(N/2)
    labels = np.concatenate((labels_1, labels_min_1))

    out_file = open("svm_feats/svm_feats_n_"+str(N)+".svm", "w")

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
