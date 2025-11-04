import faiss
import h5py
import numpy as np

def evaluate_hnsw():
    dataset_path = "sift-128-euclidean.hdf5"

    with h5py.File(dataset_path, "r") as f:
        xb = f["train"][:]   
        xq = f["test"][:]   
    print("Train vectors:", xb.shape, " Test vectors:", xq.shape)

    d = xb.shape[1]          
    M = 16
    ef_construction = 200
    ef_search = 200

    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    print("Building index ...")
    index.add(xb)
    print("Index built with {} vectors".format(index.ntotal))

    query_vec = xq[0:1]     
    D, I = index.search(query_vec, 10)  

    np.savetxt("output.txt", I[0], fmt="%d")
    print("Top-10 nearest neighbor indices written to output.txt")
    


if __name__ == "__main__":
        evaluate_hnsw()
