
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# import concurrent.futures
import multiprocessing
import numpy as np
import sys
from numpy.core.fromnumeric import nonzero
import scipy.sparse
import scipy
from scipy import io
from scipy.sparse import *
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from scipy import sparse
from time import time
import math
from random import randint
import random
from memory import *
from utils import *
from pe import *
from hashpe import *
from newhashpe import *
from fullhash import *
import numpy.ma as ma

path_prefix = "~/hash_spmm/"



def estimate_merge_task(csr_ins, curr_idx_list, used_list, n_inputs, n_pes, n_radix):
    n_depth = len(used_list) - 1
    #n_nodes = pow(n_radix, n_depth-1) + used_list[-1]
    n_nodes = np.sum(used_list)
    inner_nodes = n_nodes - n_inputs
    node_status = np.zeros((n_nodes, 3)).astype(int)  # nnzs, done, assigned
    node_status[inner_nodes:, 1] = 1

    curr_idx_list # = select_row_ids(csr_ins, out_row_id)
    assert n_inputs == curr_idx_list.size
    for i in range(n_inputs):
        node_status[inner_nodes + i, 0] = select_row_ids(csr_ins, curr_idx_list[i]).size
    
    #exame 
    wall_clk = 0
    busy_pe = 0
    pe_status = np.zeros((n_pes, 4)).astype(int) # idle|busy, id, expected_time, nnz
    while 1:
        for node_id in range(inner_nodes):
            if node_status[node_id, 2] == 0: #not assigned
                is_ready, clk_cycle_needed = check_ready(node_id, node_status, n_radix)
                if is_ready and busy_pe < n_pes:
                    node_status[node_id, 2] = 1
                    selected_pe_id = (pe_status[:, 0] == 0).nonzero()[0][0]
                    pe_status[selected_pe_id, 0] = 1
                    pe_status[selected_pe_id, 1] = node_id
                    pe_status[selected_pe_id, 2] = wall_clk + clk_cycle_needed
                    pe_status[selected_pe_id, 3] = clk_cycle_needed
                    busy_pe += 1


        # update clk
        if busy_pe > 0:
            done_pe_time = np.min( pe_status[pe_status[:, 0] != 0, 2])
            wall_clk = done_pe_time  
            pe_id = np.argmin(ma.masked_where(pe_status[:, 0]==0, pe_status[:, 2]))
            finished_node_id = pe_status[pe_id, 1]
            node_status[finished_node_id, 0] = pe_status[pe_id, 3]
            node_status[finished_node_id, 1] = 1

            pe_status[pe_id, 0] = 0
            pe_status[pe_id, 1] = 0
            pe_status[pe_id, 2] = 0
            pe_status[pe_id, 3] = 0
            busy_pe -= 1

        if node_status[:, 1].sum() == node_status.shape[0]:
            #print("done")
            break
    return wall_clk

def get_compute_cycle(i, csr_ins, N_PES, N_RADIX):
    curr_row_a = select_row_ids(csr_ins, i)
    n_inputs = curr_row_a.size
    if n_inputs > 0 :
        used_list = build_tree(n_inputs, N_RADIX)
        compute_cycle = estimate_merge_task(csr_ins, curr_row_a, used_list, n_inputs, N_PES, N_RADIX)
    return compute_cycle

def get_loading_cycle(idx, LOAD_NNZ_TRUTH, nnz_per_cycle=16):
    total_nnz = LOAD_NNZ_TRUTH[idx]
    return math.ceil(total_nnz/nnz_per_cycle)
    

if __name__ == "__main__":
    outfilename = "compute_comparision.log"
    stats_file = open(outfilename, "a+")
    csr_ins, filename = load_sparse_matrix(path_prefix)
    row_length_array = csr_ins.indptr[1:] - csr_ins.indptr[:-1]
    #total_nnz = cal_all_nnzs(csr_ins, csr_ins)

    N_PE = 32
    N_RADIX = 64
    S_OCM = 2*1024*1024 # Bytes
    clk = 0
    n_pes = 32
    n_fetcher = 16
    m_rank = csr_ins.shape[0]
    compute_cycles = np.zeros(m_rank)
    LOAD_NNZ_TRUTH = np.zeros(m_rank)
    for i in range(m_rank):
        LOAD_NNZ_TRUTH[i] = cal_nnz(csr_ins, i)
    
    print("Done loading...")
    np.argmin(LOAD_NNZ_TRUTH)
    N_PES = 32
    N_RADIX = 64
    '''
    for i in range(m_rank):
        curr_row_a = select_row_ids(csr_ins, i)
        n_inputs = curr_row_a.size
        if n_inputs > 0 :
            used_list = build_tree(n_inputs, N_RADIX)
            compute_cycles[i] = estimate_merge_task(csr_ins, curr_row_a, used_list, n_inputs, N_PES, N_RADIX)
    '''
    min_idx = np.argmin(ma.masked_where(row_length_array<10, row_length_array))
    max_idx = np.argmax(ma.masked_where(row_length_array<10, row_length_array))

    min_compute_cycle = get_compute_cycle(min_idx, csr_ins, N_PES, N_RADIX)
    min_load_cycle = get_loading_cycle(min_idx, LOAD_NNZ_TRUTH)
    max_compute_cycle = get_compute_cycle(max_idx, csr_ins, N_PES, N_RADIX)
    max_load_cycle = get_loading_cycle(max_idx, LOAD_NNZ_TRUTH)

    print("{} {} {} {} {} {} {} {}".format(filename, csr_ins.nnz, csr_ins.nnz/csr_ins.shape[0], \
        np.max(row_length_array), min_compute_cycle, min_load_cycle, max_compute_cycle, max_load_cycle), file=stats_file)

        