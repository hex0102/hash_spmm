import scipy
from scipy import io
import numpy as np
import sys
import math
import concurrent.futures
import threading
import time
from utils import *



def count_N_summation(input_a, input_b):
    collision_free_percentage = np.zeros(input_a.shape[0])
    for i in range(input_a.shape[0]):
        column_ids = getNNZ_by_id(input_a, input_b, i)
        if column_ids.size>=4096:
            count_freq = np.bincount(column_ids)
            #total_nnz_row = column_ids.size  #(count_freq).sum()
            nonzero_idx = count_freq.nonzero()[0]
            total_nnz_row = nonzero_idx.size
            no_collision_nnz = ( (count_freq <= 4)*(count_freq > 0)).nonzero()[0].size
            small_amount = count_freq[((count_freq <= 4)*(count_freq > 0)).nonzero()[0]].sum()

            collision_free_percentage[i] = small_amount/column_ids.size
    return collision_free_percentage





def get_gt_way(input_a, n_way):
    input_m = input_a.tocsr()
    gt_cc = 0
    for i in range(input_m.shape[0]):
        nnz_i = input_m.indptr[i+1] - input_m.indptr[i]
        if nnz_i>n_way:
            gt_cc+=1
    return gt_cc

def get_median_max(input_a, input_b, partition_size):
    input_a = input_a.tocsr()
    input_b = input_b.tocsr()
    flops_row = np.zeros(input_a.shape[0])
    for i in range(input_a.shape[0]):
        col_ids = select_row_ids(input_a, i)
        row_lens = get_row_lengths(input_b, col_ids)
        flops_row[i] = np.sum(row_lens)


    diff_list = np.zeros(math.ceil(flops_row.size/partition_size))
    n_partition = math.ceil(flops_row.size/partition_size)

    max_nnz = 0
    average_nnz = 0
    for i in range(n_partition):
        curr_partition = flops_row[i*partition_size:(i + 1)*partition_size]
        max_flops = np.max(curr_partition)
        median_flops = np.median(curr_partition)
        if(median_flops!=0):
            diff_list[i] = (max_flops - median_flops)/median_flops
        else:
            diff_list[i] = 0

        max_nnz += max_flops
        average_nnz += int(np.mean(curr_partition))
    return max_nnz, average_nnz


if __name__ == "__main__":
    print_freq = 5000
    stats_file = open("load_balance_stats.txt", "a+")
    path_prefix = "/home/xinhe/hash_spmm/inputs/"
    full_path = path_prefix + sys.argv[1]
    input_a = scipy.io.mmread(full_path) #"filter3D/filter3D.mtx"
    #'consph/consph.mtx'
    #'cop20k_A/cop20k_A.mtx'
    #'hood/hood.mtx'
    #'mono_500Hz/mono_500Hz.mtx'
    #'pdb1HYS/pdb1HYS.mtx'
    #'webbase-1M/webbase-1M.mtx'
    input_a = input_a.tocsr()
    output_c = input_a*input_a
    no_collision_percentage = count_N_summation(input_a, input_a)
    data_name = full_path.rsplit('/', 1)[-1]
    print("A_NNZ:{} A_NNZ/ROW:{} C_NNZ:{} C_NNZ/ROW:{}".format(input_a.nnz, input_a.nnz/input_a.shape[0], output_c.nnz, output_c.nnz/output_c.shape[0]))
    n_row = input_a.shape[0]
    n_col = input_a.shape[1]
    assert(input_a.shape[0] == input_a.shape[1])
    input_a = input_a.tocsr()
    
    gt_cc = get_gt_way(input_a, 64)
    max_nnz, average_nnz = get_median_max(input_a, input_a, 512)

    #print("{},{},{},{},{},{},{}".format(data_name, input_a.shape[0], input_a.nnz, input_a.nnz/input_a.shape[0], 
    #    output_c.nnz, output_c.nnz/output_c.shape[0], gt_cc), file=stats_file)

    print("{},{},{},{},{},{},{},{},{},{},{}".format(data_name, input_a.shape[0], input_a.nnz, input_a.nnz/input_a.shape[0], 
        output_c.nnz, output_c.nnz/output_c.shape[0], gt_cc, gt_cc/input_a.shape[0], max_nnz, average_nnz, (max_nnz-average_nnz)/average_nnz), file=stats_file)

    print("Completed")