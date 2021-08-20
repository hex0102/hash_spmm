import scipy
from scipy import io
import numpy as np
import sys
import math
import concurrent.futures
import threading
import time

def select_row_ids(csr_ins, row_id):
    return csr_ins.indices[csr_ins.indptr[row_id] : csr_ins.indptr[row_id + 1]]

def estimate_inner_p(input_m):
    input_a = input_m.tocsr()
    input_b = input_m.tocsc()
    total_read = 0
    total_write = 0
    
    '''
    for i in range(input_a.shape[0]):
        total_read += 2
        row_length = input_a.indptr[i + 1] - input_a.indptr[i]
        if row_length != 0:
            for j in range(input_b.shape[1]):
                total_read += 2
                col_length = input_b.indptr[j + 1] - input_b.indptr[j]
                if col_length != 0: 
                    total_read += row_length + col_length
                    total_write += 1
    '''
    col_length_list = input_b.indptr[1 : input_b.shape[1] + 1] - input_b.indptr[: input_b.shape[1]]
    nonzero_col_size = col_length_list.nonzero()[0].size 
    
    # read input_a
    total_read += input_a.shape[0] # read indptr
    total_read += input_a.nnz * nonzero_col_size

    # read input_b
    total_read += input_a.shape[0]*input_b.shape[1] # read indptr
    total_read += input_a.shape[0]*input_b.nnz

    total_write += (input_a*input_b).nnz

    return total_read, total_write


def estimate_outer_p(input_m, num_way):
    input_a = input_m.tocsc()
    input_b = input_m.tocsr()
    total_read = 0
    total_write = 0

    #mult phase
    partial_results_length = []
    for i in range(input_a.shape[1]):
        total_read += 2 + 2
        col_length = input_a.indptr[i + 1] - input_a.indptr[i]
        row_length = input_b.indptr[i + 1] - input_b.indptr[i]
        
        if col_length != 0 and row_length != 0:
            total_read += col_length*row_length
            total_write += col_length*row_length
            partial_results_length.append(col_length*row_length)
        
    partial_results_length.sort()
    #merge phase
    n_partial_results = len(partial_results_length)
    N_STAGE = math.ceil(math.log(n_partial_results)/math.log(num_way))

    assert(N_STAGE==2)
    y = math.ceil((n_partial_results - pow(num_way, N_STAGE-1))/(num_way - 1))
    x = pow(num_way, N_STAGE-1) - y

    #first stage    
    for i in range(y):
        if((i+1)*num_way< n_partial_results- x):
            curr_merge = partial_results_length[i*num_way : (i+1)*num_way]
        else:
            curr_merge = partial_results_length[i*num_way : n_partial_results - x]
        total_read += sum(curr_merge)
        total_write += sum(curr_merge)

    #scond stage
    for i in range(y):
        if((i+1)*num_way< n_partial_results- x):
            curr_merge = partial_results_length[i*num_way : (i+1)*num_way]
        else:
            curr_merge = partial_results_length[i*num_way : n_partial_results - x]

        total_read += sum(curr_merge)
        total_write += sum(curr_merge)
    
    total_read += sum(partial_results_length[x:])
    total_write += sum(partial_results_length[x:])

    return total_read, total_write


def estimate_outer_p_hash(input_m):
    input_a = input_m.tocsc()
    input_b = input_m.tocsr()
    total_read = 0
    total_write = 0

    #mult phase
    for i in range(input_a.shape[1]):
        total_read += 2 + 2
        col_length = input_a.indptr[i + 1] - input_a.indptr[i]
        row_length = input_b.indptr[i + 1] - input_b.indptr[i]
        
        if col_length != 0 and row_length != 0:
            total_read += col_length*row_length
            total_write += col_length*row_length
            #partial_results_length.append(col_length*row_length)
        
    #merge phase
    

    return total_read, total_write


def estimate_row_wise_p(input_m):
    # we assume a merge tree with enough number of ways since row-wise method only merges a limited number of partial lists
    input_a = input_m.tocsr()
    input_b = input_m.tocsr()
    total_read = 0
    total_write = 0

    input_c = input_a*input_b

    '''
    for i in range(input_a.shape[0]):
        total_read += 1
        row_start = input_a.indptr[i]
        row_end = input_a.indptr[i + 1]
        row_length = row_end- row_start
        select_row_lists = []
        if row_length != 0:
            for j in range(row_start, row_end):
                col_id = input_a.indices[j]
                selected_row = select_row_ids(input_b, col_id)
                if(selected_row.size!=0):
                    select_row_lists.append(selected_row)
                    # read/write when reading a value from matrix a and multiplying it with a row from matrix b 
                    total_read += selected_row.size 
                    total_write += selected_row.size
            union_list = np.array([])
            for j in range(len(select_row_lists)):
                total_read += select_row_lists[j].size
                union_list = np.union1d(union_list, select_row_lists[j])
            total_write += union_list.size
    
    print("{} {}".format(total_read, total_write))
    '''

    total_read = 0
    total_write = 0
    total_read += input_a.shape[0]
    total_read += input_a.indices.size
    selected_rows = input_b[input_a.indices]
    total_read += selected_rows.nnz
    total_write += selected_rows.nnz
    
    total_read += selected_rows.nnz
    total_write += input_c.nnz

    print("{} {}".format(total_read, total_write))

    return total_read, total_write


def estimate_row_wise_p_hash(input_m):
    input_a = input_m.tocsr()
    input_b = input_m.tocsr()
    total_read = 0
    total_write = 0

    input_c = input_a*input_b

    total_read = 0
    total_write = 0
    total_read += input_a.shape[0] # indptr array of matrix a
    total_read += input_a.indices.size # indices array of matrix a 
    selected_rows = input_b[input_a.indices]
    total_read += 2*selected_rows.nnz # using bloom filter can help reduce the amount of checkup
    total_write += selected_rows.nnz
    
    #total_read += selected_rows.nnz
    #total_write += input_c.nnz

    print("{} {}".format(total_read, total_write))

    return total_read, total_write



if __name__ == "__main__":
    print_freq = 5000
    stats_file = open("dataflow_stats.txt", "a+")
    path_prefix = "/home/xinhe/hash_spmm/"
    full_path = path_prefix + sys.argv[1]
    input_a = scipy.io.mmread(full_path) #"filter3D/filter3D.mtx"
    output_c = input_a*input_a
    
    data_name = full_path.rsplit('/', 1)[-1]
    print("A_NNZ:{} A_NNZ/ROW:{} C_NNZ:{} C_NNZ/ROW:{}".format(input_a.nnz, input_a.nnz/input_a.shape[0], output_c.nnz, output_c.nnz/output_c.shape[0]))
    n_row = input_a.shape[0]
    n_col = input_a.shape[1]
    assert(input_a.shape[0] == input_a.shape[1])
    
    

    #inner estimate
    print("computing inner prod...")
    ip_reads0, ip_writes0 = estimate_inner_p(input_a)

    #outer estimate
    print("computing outer prod...")
    ip_reads1, ip_writes1 = estimate_outer_p(input_a, 1024)

    print("computing outer prod hash...")
    ip_reads2, ip_writes2 = estimate_outer_p_hash(input_a)

    #row-wise / gustavson estimate
    print("computing row-wise...")
    ip_reads3, ip_writes3 = estimate_row_wise_p(input_a)

    print("computing row-wise hash...")
    ip_reads4, ip_writes4 = estimate_row_wise_p_hash(input_a)



    print("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(data_name, input_a.shape[0], input_a.nnz, input_a.nnz/input_a.shape[0], 
        output_c.nnz, output_c.nnz/output_c.shape[0], ip_reads0, ip_writes0, ip_reads1, ip_writes1, ip_reads2, ip_writes2, ip_reads3, ip_writes3, ip_reads4, ip_writes4), file=stats_file)

    print("finished")