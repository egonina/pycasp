import numpy as np

D = 19 
Ns = [1000000, 5000000, 10000000]

for N in Ns:
    feats = np.random.randn(N, D)

    out_file = open("gmm_sample_data/gmm_feats_n_"+str(N)+".gmm", "w")

    row_count = 0
    for i in range(N):
        for j in range(D):
            if j < D-1:
                out_file.write(str(feats[i][j])+" ")
            else:
                out_file.write(str(feats[i][j])+"\n")

    out_file.close()
