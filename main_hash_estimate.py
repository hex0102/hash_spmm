import scipy
from scipy import io
import numpy as np
import sys
import concurrent.futures
import threading
import time
import math
from utils import *

vectorize = 0
V_LEN = 4

HASH_SCAL = 107
MIN_HT_S = 8
print_freq = 10000

def select_row_ids(csr_ins, row_id):
    return csr_ins.indices[csr_ins.indptr[row_id] : csr_ins.indptr[row_id + 1]]

def select_row_vals(csr_ins, row_id):
    return csr_ins.data[csr_ins.indptr[row_id] : csr_ins.indptr[row_id + 1]]

def hash_symbolic_hash(csr_a, csr_b, ht_size_array):
    c_nnz = np.zeros(csr_a.shape[0])
    for i in range(csr_a.shape[0]):
        if i % print_freq == 0:
            print(i)
        curr_col_ids = csr_a.indices[csr_a.indptr[i] : csr_a.indptr[i+1]]
        curr_vals = csr_a.val[csr_a.indptr[i] : csr_a.indptr[i+1]]
        cur_nnz = 0
        # process NZs in the row of A matrix
        for j in range(curr_col_ids.size):
            cur_row_ids_b = select_row_ids(csr_b, curr_col_ids[j])
            cur_row_vals_b = select_row_vals(csr_b, curr_col_ids[j])
            t_aval = curr_vals[j]
            # process NZs in the row of B matrix
            for m in range(cur_row_ids_b.size):
                key = cur_row_ids_b[m]
                b_val = cur_row_vals_b[m]
                hash_addr = (key * HASH_SCAL) & (ht_size_array[i] - 1)
                while(1):
                    break
    return c_nnz


#0 if x == 0 else 2**(x - 1).bit_length()
def next_power_of_2(x):
    ht_size_array = np.zeros(x.size).astype(int)
    x = x.astype(int)
    for i in range(x.size):
        ht_size_array[i] =  0 if x[i] == 0 else 2**(int(x[i]) - 1).bit_length() 
    return ht_size_array


#the nnz is calculated, instead of hashed
def hash_symbolic_calculated(csr_a, csr_b):
    csr_a = csr_a.tocsr()
    csr_b = csr_b.tocsr()
    c_nnz = np.zeros(csr_a.shape[0])
    total_multiplication = 0
    for i in range(csr_a.shape[0]):
        if i % print_freq == 0:
            print(i)
        c_row = np.array([])
        curr_col_ids = csr_a.indices[csr_a.indptr[i] : csr_a.indptr[i+1]]
        for j in range(curr_col_ids.size):
            selected_row_b = select_row_ids(csr_b, curr_col_ids[j])
            total_multiplication += selected_row_b.size
            c_row = np.union1d(c_row, selected_row_b)
        c_nnz[i]=c_row.size      
    return c_nnz

#the nnz is calculated, instead of hashed
def hash_symbolic_upperbound(csr_a, csr_b):
    csr_a = csr_a.tocsr()
    csr_b = csr_b.tocsr()
    c_nnz = np.zeros(csr_a.shape[0])
    for i in range(csr_a.shape[0]):
        if i % print_freq == 0:
            print(i)
        curr_col_ids = csr_a.indices[csr_a.indptr[i] : csr_a.indptr[i+1]]
        for j in range(curr_col_ids.size):
            c_nnz[i] += select_row_ids(csr_b, curr_col_ids[j]).size    
    return c_nnz

def hash_output_per_row(csr_a, csr_b, ht_check, ht_val, ht_size_array, i):
    #ht_check = ht_check_a[i]
    #ht_val = ht_val_a[i]
    curr_col_ids = csr_a.indices[csr_a.indptr[i] : csr_a.indptr[i+1]]
    curr_vals = csr_a.data[csr_a.indptr[i] : csr_a.indptr[i+1]]
    hash_linear_probing = 0
    # process NZs in the ith row of A matrix
    for j in range(curr_col_ids.size):
        cur_row_ids_b = select_row_ids(csr_b, curr_col_ids[j])
        cur_row_vals_b = select_row_vals(csr_b, curr_col_ids[j])
        t_aval = curr_vals[j]
        # process NZs in the row of B matrix
        for m in range(cur_row_ids_b.size):
            key = cur_row_ids_b[m]
            t_val = t_aval * cur_row_vals_b[m]
            hash_addr = (key * HASH_SCAL) & (ht_size_array - 1)
            while 1:
                if (ht_check[hash_addr] == key):
                    ht_val[hash_addr] = t_val + ht_val[hash_addr]
                    break
                elif ht_check[hash_addr] == -1:
                    ht_check[hash_addr] = key
                    ht_val[hash_addr] = t_val
                    break
                else:
                    hash_addr = (hash_addr + 1) & (ht_size_array - 1)
                    hash_linear_probing += 1
    if i % print_freq ==0:
        print("done with row {}".format(i))
    return hash_linear_probing, ht_check, ht_val

def hash_output_per_row_vec(csr_a, csr_b, ht_check, ht_val, ht_size_array, i, v_len):
    curr_col_ids = csr_a.indices[csr_a.indptr[i] : csr_a.indptr[i+1]]
    curr_vals = csr_a.data[csr_a.indptr[i] : csr_a.indptr[i+1]]
    hash_linear_probing = 0
    # process NZs in the ith row of A matrix
    for j in range(curr_col_ids.size):
        cur_row_ids_b = select_row_ids(csr_b, curr_col_ids[j])
        cur_row_vals_b = select_row_vals(csr_b, curr_col_ids[j])
        t_aval = curr_vals[j]
        # process NZs in the row of B matrix
        for m in range(cur_row_ids_b.size):
            key = cur_row_ids_b[m]
            t_val = t_aval * cur_row_vals_b[m]
            hash_addr = (key * HASH_SCAL) & ((ht_size_array>>int(math.log2(v_len))) - 1)
            while 1:
                flag = 0
                for l in range(v_len):
                    if (ht_check[hash_addr*v_len + l] == key):
                        ht_val[hash_addr*v_len + l] = t_val + ht_val[hash_addr*v_len + l]
                        flag = 1
                        break
                if flag == 1:
                    break
                empty_slot = (ht_check[hash_addr*v_len: hash_addr*v_len+v_len]==-1).nonzero()[0]
                
                if(empty_slot.size != 0):
                    ht_check[hash_addr*v_len +  empty_slot[0]] = key
                    ht_val[hash_addr*v_len +  empty_slot[0]] = t_val
                    break
                else:
                    hash_addr = (hash_addr + 1) & ((ht_size_array>>int(math.log2(v_len))) - 1)
                    hash_linear_probing += 1
                    
    if i % print_freq ==0:
        print("done with row {}".format(i))
    return hash_linear_probing, ht_check, ht_val


def hash_numeric_parallel(csr_a, csr_b, ht_size_array):
    start_time = time.time()
    ht_check = [np.array([]).astype(int) for _ in range(csr_a.shape[0])]
    ht_val = [np.array([]) for _ in range(csr_a.shape[0])]
    
    for i in range(ht_size_array.size):
        if(ht_size_array[i]>0 and ht_size_array[i]<V_LEN):
            ht_size_array[i] = V_LEN
        ht_check[i].resize(int(ht_size_array[i]))
        ht_check[i].fill(-1)
        ht_val[i].resize(int(ht_size_array[i]))
        ht_val[i].fill(0)
    
    if not vectorize:
        print("not vectorized")
    else:
        print("vectorized with len: {}".format(V_LEN))

    total_collision = 0
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for i in range(csr_a.shape[0]):
            if(not vectorize):
                future = executor.submit(hash_output_per_row, csr_a, csr_b, ht_check[i], ht_val[i], ht_size_array[i], i)   
            else:
                future = executor.submit(hash_output_per_row_vec, csr_a, csr_b, ht_check[i], ht_val[i], ht_size_array[i], i, V_LEN)
            #print("--- %s finished ---" % (i))
            futures.append(future)
    
    for i in range(len(futures)):
        total_collision += futures[i].result()[0]
        ht_check[i] = futures[i].result()[1]
        ht_val[i] = futures[i].result()[2]
    print("--- %s seconds ---" % (time.time() - start_time))
    print(total_collision)
    return total_collision, ht_check, ht_val


def hash_numeric(csr_a, csr_b, ht_size_array):
    print("hash_numeric()...")
    ht_check = [np.array([]).astype(int) for _ in range(csr_a.shape[0])]
    ht_val = [np.array([]) for _ in range(csr_a.shape[0])]
    for i in range(ht_size_array.size):
        if(ht_size_array[i]>0 and ht_size_array[i]<V_LEN):
            ht_size_array[i] = V_LEN        
        ht_check[i].resize(int(ht_size_array[i]))
        ht_check[i].fill(-1)
        ht_val[i].resize(int(ht_size_array[i]))
        ht_val[i].fill(0)
    hash_linear_probing = 0
    start_time = time.time()

    if not vectorize:
        print("not vectorized")
    else:
        print("vectorized with len: {}".format(V_LEN))

    for i in range(csr_a.shape[0]):
        #if i % print_freq == 0:
        #    print(i)
        if not vectorize:
            curr_col_ids = csr_a.indices[csr_a.indptr[i] : csr_a.indptr[i+1]]
            curr_vals = csr_a.data[csr_a.indptr[i] : csr_a.indptr[i+1]]
            cur_nnz = 0
            
            # process NZs in the row of A matrix
            for j in range(curr_col_ids.size):
                cur_row_ids_b = select_row_ids(csr_b, curr_col_ids[j])
                cur_row_vals_b = select_row_vals(csr_b, curr_col_ids[j])
                t_aval = curr_vals[j]
                # process NZs in the row of B matrix
                for m in range(cur_row_ids_b.size):
                    key = cur_row_ids_b[m]
                    t_val = t_aval *cur_row_vals_b[m]
                    hash_addr = (key * HASH_SCAL) & (ht_size_array[i] - 1)
                    while 1:
                        if (ht_check[i][hash_addr] == key):
                            ht_val[i][hash_addr] = t_val + ht_val[i][hash_addr]
                            break
                        elif ht_check[i][hash_addr] == -1:
                            ht_check[i][hash_addr] = key
                            ht_val[i][hash_addr] = t_val
                            break
                        else:
                            hash_addr = (hash_addr + 1) & (ht_size_array[i] - 1)
                            hash_linear_probing+=1
        else:
            total_collision, ht_check[i], ht_val[i] = hash_output_per_row_vec(csr_a, csr_b, ht_check[i], ht_val[i], ht_size_array[i], i, V_LEN)
            hash_linear_probing+=total_collision

    print(hash_linear_probing)
    print("--- %s seconds ---" % (time.time() - start_time))
    return ht_check, ht_val

def report_occupancy(csr_a, csr_b, ht_size_array):
    csr_c = csr_a * csr_b
    row_length_array = csr_c.indptr[1:] - csr_c.indptr[:-1]

    non_empty_inx = np.nonzero(ht_size_array)
    real_occupancy = row_length_array[non_empty_inx]/ht_size_array[non_empty_inx]
    return real_occupancy.mean()



#this script compares on-chip read and write  of matraptor and hash-based method, assuming enough on-chip memory
if __name__ == "__main__":
    print_freq = 10000
    outfilename = "init_collision_comparision.log"
    stats_file = open(outfilename, "a+")
    path_prefix = "/home/xinhe/hash_spmm/"
    input_a, filename = load_sparse_matrix(path_prefix)
    '''
    full_path = path_prefix + sys.argv[1]
    input_a = scipy.io.mmread(full_path) #"filter3D/filter3D.mtx"
    '''
    output_c = input_a*input_a
    
    print("A_NNZ:{} A_NNZ/ROW:{} C_NNZ:{} C_NNZ/ROW:{}".format(input_a.nnz, input_a.nnz/input_a.shape[0], output_c.nnz, output_c.nnz/output_c.shape[0]))
    n_row = input_a.shape[0]
    n_col = input_a.shape[1]
    assert(input_a.shape[0] == input_a.shape[1])
    input_a = input_a.tocsr()
    c_nnz1 = hash_symbolic_calculated(input_a, input_a)
    total_nnz = cal_all_nnzs(input_a, input_a)

    # c_nnz2 = hash_symbolic_upperbound(input_a, input_a)
    ht_size_array = next_power_of_2(c_nnz1)
    # ht_check0, ht_val0 = hash_numeric(input_a, input_a, ht_size_array)
    total_collision, ht_check, ht_val = hash_numeric_parallel(input_a, input_a, ht_size_array)
    row_length_array = input_a.indptr[1:] - input_a.indptr[:-1]

    occupancy = report_occupancy(input_a, input_a, ht_size_array) 
    print("{} {} {} {} {} {} {} {}".format(filename, input_a.nnz, input_a.nnz/input_a.shape[0], \
        np.max(row_length_array), total_nnz, total_nnz/input_a.shape[0], total_collision, occupancy), file=stats_file)
    #print(c_nnz2.sum()) 
    print("Completed")
