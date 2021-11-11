
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


def cal_working_set(curr_row_idx, N_PES, LOAD_NNZ_TRUTH):
    footprint = 0
    for i in range(N_PES):
        if(curr_row_idx[i]!=-1):
            footprint += LOAD_NNZ_TRUTH[i]
    return footprint

if __name__ == "__main__":
    outfilename = "bandwidth_comparision.log"
    stats_file = open(outfilename, "a+")
    csr_ins, filename = load_sparse_matrix(path_prefix)
    row_length_array = csr_ins.indptr[1:] - csr_ins.indptr[:-1]
    #total_nnz = cal_all_nnzs(csr_ins, csr_ins)

    N_PE = 32
    N_RADIX = 64
    S_OCM = 2*1024*1024 # Bytes
    clk = 0
    n_pes = 16
    n_fetcher = 16
    m_rank = csr_ins.shape[0]
    load_nnz = np.zeros(m_rank)
    print("Done loading...")
    LOAD_NNZ_TRUTH = np.zeros(m_rank)
    for i in range(m_rank):
        LOAD_NNZ_TRUTH[i] = cal_nnz(csr_ins, i)

    processed = np.zeros(csr_ins.shape[0])
    done  = np.zeros(csr_ins.shape[0])

    OCM_SIZE = 1*1024*1024/8
    remaining = OCM_SIZE

    load_set = list(range(n_fetcher))
    next_row_id = 16
    cc_loader = 0

    clk = 0

    pe_row_id = list(range(n_pes))
    next_processing_id = 16
    
    write_n = np.zeros(csr_ins.shape[0])
    finished = np.zeros(csr_ins.shape[0])

    print("start estimation")


    total_r_nnz = 0
    total_w_nnz = 0
    mode = 1
 
    # complete decoupling of fetching and processing
    if mode == 0:
        for i in range(m_rank):
            if LOAD_NNZ_TRUTH[i] == 0:
                finished[i] = 1
        while(1):
            nonzero_list = (finished == 0).nonzero()[0]
            n_local = 0
            
            # write the oldest output row
            if nonzero_list.size != 0:
                earliest_idx = nonzero_list[0]
                if finished[earliest_idx] == 0:
                    if processed[earliest_idx] - write_n[earliest_idx] >= 16 or ( processed[earliest_idx] - write_n[earliest_idx]< 16  and processed[earliest_idx] == LOAD_NNZ_TRUTH[earliest_idx] ):
                        n_local = processed[earliest_idx] - write_n[earliest_idx]
                        if n_local > 16:
                            n_local = 16
                        write_n[earliest_idx] += n_local
                        remaining += n_local
                        total_w_nnz += n_local
                    if( write_n[earliest_idx]>=LOAD_NNZ_TRUTH[earliest_idx] ):
                        finished[earliest_idx] = 1
                        print("row {} finished".format(earliest_idx))

            # fetch 
            if remaining > 16 and n_local == 0:
                selected_row = load_set[cc_loader]
                if selected_row < m_rank:
                    load_nnz[selected_row] += 16
                    if(load_nnz[selected_row] >= LOAD_NNZ_TRUTH[selected_row]):
                        load_nnz[selected_row] = LOAD_NNZ_TRUTH[selected_row]
                        load_set[cc_loader] = next_row_id
                        next_row_id +=1
                        remaining -= 16 - (load_nnz[selected_row] - LOAD_NNZ_TRUTH[selected_row])
                        total_r_nnz += 16 - (load_nnz[selected_row] - LOAD_NNZ_TRUTH[selected_row])
                    else:
                        remaining -= 16
                        total_r_nnz += 16
                cc_loader += 1
                if cc_loader == n_fetcher:
                    cc_loader = 0

            
            #start processing when there is data
            for i in range(n_pes):
                curr_row_id = pe_row_id[i]
                if curr_row_id < m_rank:
                    if processed[curr_row_id] < load_nnz[curr_row_id]:
                        processed[curr_row_id] += 1 
                    
                    if processed[curr_row_id] >= LOAD_NNZ_TRUTH[curr_row_id]:
                        done[pe_row_id[i]] = 1 # done processing
                        pe_row_id[i] = next_processing_id
                        next_processing_id += 1



            if np.sum(finished) == m_rank:
                print("done")
                break

            clk += 1
    elif mode == 1:

        row_idx_ptr = 0
        curr_row_idx = np.ones(n_pes).astype(int) * -1 # the row id the pe is processing
        curr_row_done = np.ones(n_pes) # if the correpsonding pe finished processing

        while 1:
            n_local = 0
            
            earliest_idx = -1
            pe_id = -1
            if curr_row_idx[curr_row_idx!=-1].size:
                earliest_idx = np.min(curr_row_idx[curr_row_idx!=-1])
                pe_id = np.argmin(ma.masked_where(curr_row_idx==-1, curr_row_idx))  
                #np.argmin(curr_row_idx[curr_row_idx!=-1]) + np.sum(curr_row_idx==-1)
            if earliest_idx != -1:
                if processed[earliest_idx] - write_n[earliest_idx] >= 16 or (processed[earliest_idx] - write_n[earliest_idx]< 16 and processed[earliest_idx] - write_n[earliest_idx] > 0 and processed[earliest_idx] == LOAD_NNZ_TRUTH[earliest_idx] ):
                    n_local = processed[earliest_idx] - write_n[earliest_idx]
                    if n_local > 16:
                        n_local = 16
                    write_n[earliest_idx] += n_local
                    remaining += n_local
                    total_w_nnz += n_local
                
                if( write_n[earliest_idx]>=LOAD_NNZ_TRUTH[earliest_idx] ):
                    finished[earliest_idx] = 1
                    curr_row_idx[pe_id] = -1
                    curr_row_done[pe_id] = 1
                    print("row {} finished".format(earliest_idx))
                    #print("remaining {}".format(remaining))        
            
            #start processing when there is data
            for i in range(n_pes):
                selected_row_id = curr_row_idx[i]
                if selected_row_id < m_rank and selected_row_id != -1:
                    if processed[selected_row_id] < load_nnz[selected_row_id]:
                        processed[selected_row_id] += 1 
                    
                    if processed[selected_row_id] >= LOAD_NNZ_TRUTH[selected_row_id]:
                        done[selected_row_id] = 1 # done processing
            
            # fetch into load_nnz
            load_n = 0
            if remaining > 16 and n_local == 0:
                for cc in range(n_pes):
                    selected_row = curr_row_idx[cc_loader]
                    if selected_row < m_rank and selected_row !=-1:
                        if load_nnz[selected_row] < LOAD_NNZ_TRUTH[selected_row]:
                            load_nnz[selected_row] += 16
                            if(load_nnz[selected_row] >= LOAD_NNZ_TRUTH[selected_row]):
                                remaining -= 16 - (load_nnz[selected_row] - LOAD_NNZ_TRUTH[selected_row])
                                total_r_nnz += 16 - (load_nnz[selected_row] - LOAD_NNZ_TRUTH[selected_row])
                                load_nnz[selected_row] = LOAD_NNZ_TRUTH[selected_row]
                            else:
                                remaining -= 16
                                total_r_nnz += 16
                            load_n = 1
                    
                    cc_loader += 1
                    if cc_loader == n_fetcher:
                        cc_loader = 0
                    if load_n == 1:
                        break
            

            
            # scheduler
            for i in range(n_pes):
                if curr_row_done[i] == 1:
                    used_space = cal_working_set(curr_row_idx, n_pes, LOAD_NNZ_TRUTH)
                    if row_idx_ptr < m_rank:
                        if used_space + LOAD_NNZ_TRUTH[row_idx_ptr] <= OCM_SIZE or ( used_space == 0 ):
                            curr_row_idx[i] = row_idx_ptr
                            row_idx_ptr += 1
                            curr_row_done[i] = 0 
                    else:
                        break
            
            
            if np.sum(finished) == m_rank:
                print("done")
                break
            
            clk += 1
            if clk%100000 == 0:
                print( "{} GB/cycle".format((total_r_nnz+total_w_nnz)*8/clk))
    else:
        print("unknown mode")
        
    bandwidth = (total_r_nnz+total_w_nnz)*8/clk

    print("{} {} {} {} {}".format(filename, csr_ins.nnz, csr_ins.nnz/csr_ins.shape[0], \
        np.max(row_length_array), bandwidth), file=stats_file)
        