import scipy
from scipy import io
import numpy as np
import sys

#this script compares the allocation of intermeidate results

if __name__ == "__main__":

    print_freq = 5000
    stats_file = open("rowwise_space.txt", "a+")
    path_prefix = "/home/xinhe/hash_spmm/inputs/"
    full_path = path_prefix + sys.argv[1]
    input_a = scipy.io.mmread(full_path) #"filter3D/filter3D.mtx"
    input_csr = input_a.tocsr()
    row_length_array = input_csr.indptr[1:] - input_csr.indptr[:-1]

    output_c = input_a*input_a

    data_name = full_path.rsplit('/', 1)[-1]
    print("A_NNZ:{} A_NNZ/ROW:{}  MAX_A_ROW:{} C_NNZ:{} C_NNZ/ROW:{}".format(input_a.nnz, input_a.nnz/input_a.shape[0], \
        np.max(row_length_array), output_c.nnz, output_c.nnz/output_c.shape[0]))
    print("{} {} {} {} {} {}".format(data_name, input_a.nnz, input_a.nnz/input_a.shape[0], \
        np.max(row_length_array), output_c.nnz, output_c.nnz/output_c.shape[0]), file=stats_file)
