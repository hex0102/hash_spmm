# this file is the main py for the cycle accurate simulation of hash-spmm
# baseline or proposed

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# import concurrent.futures
import multiprocessing
import numpy as np
import sys
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

path_prefix = "~/hash_spmm/"

# hyper-params
n_channels = 8 # number of memory channels 
n_pes = 16


def start_run(csr_ins):
    # setup
    csr_out = csr_ins*csr_ins
    matrix_space_bound = csr_ins.shape[0]*4 + csr_ins.nnz*8 
    N_ROWS = csr_ins.shape[0]
    main_memory = Memory(n_channels, matrix_space_bound, N_ROWS)
    pe_array = []
    shared_complete_list = np.zeros(csr_out.shape[0])
    shared_status_table = np.zeros(n_pes)
    for i in range(n_pes):
        assigned_rows =[ *range(i, csr_ins.shape[0], n_pes) ]
        pe_array.append(BasePE(i, assigned_rows, csr_ins, csr_out, shared_complete_list, shared_status_table) )

    clk = 0    
    while(1):
        #gather returned request from all memory channels
        returned_requests = main_memory.tick()

        #process the received requests.
        for i in range(len(returned_requests)):
            curr_request = returned_requests[i]
            r_valid = curr_request[0]
            r_type = curr_request[1]
            r_id = curr_request[2]
            r_addr = curr_request[3]
            r_n = curr_request[4]
            # process valid and read request
            if(r_valid == 1 and r_type == 0): 
                pe_array[r_id].receive(curr_request)
        

        #gather the requests from pes and send it to main memory
        '''
        executor = ThreadPoolExecutor(max_workers=16)
        for result in executor.map(BasePE.tick, pe_array):
            #list_a.append(result)
            main_memory.enqueue(curr_request)
        '''
        
        for i in range(n_pes):
            curr_request = pe_array[i].tick()
            main_memory.enqueue(curr_request)
            pass
        

        if sum(shared_status_table) == n_pes:
            print("total request handled: {}".format(main_memory.n_request))
            print("The overall execution time is: "+str(clk)+" cycles.")
            return clk
            #break
        clk += 1 #increment the clk counter

def start_run_hash(csr_ins):
    # setup
    csr_out = csr_ins*csr_ins
    matrix_space_bound = csr_ins.shape[0]*4 + csr_ins.nnz*8 
    N_ROWS = csr_ins.shape[0]
    main_memory = Memory(n_channels, matrix_space_bound, N_ROWS)
    pe_array = []
    shared_complete_list = np.zeros(csr_out.shape[0])
    shared_status_table = np.zeros(n_pes)
    
    assigned_rows =[ *range(0, csr_ins.shape[0], 1) ]
    pe_array.append(HashPE(0, n_pes, assigned_rows, csr_ins, csr_out, shared_complete_list, shared_status_table) )

    clk = 0    
    while(1):
        #gather returned request from all memory channels
        returned_requests = main_memory.tick()

        #process the received requests.
        for i in range(len(returned_requests)):
            curr_request = returned_requests[i]
            r_valid = curr_request[0]
            r_type = curr_request[1]
            r_id = curr_request[2]
            r_addr = curr_request[3]
            r_n = curr_request[4]
            # process valid and read request
            if(r_valid == 1 and r_type == 0): 
                pe_array[0].receive(curr_request)
        

        #gather the requests from pes and send it to main memory
        #for i in range(n_pes):
        curr_request_list = pe_array[0].tick()
        for i in range(len(curr_request_list)):
            curr_request = curr_request_list.pop(0)
            main_memory.enqueue(curr_request)
            pass

        if sum(shared_status_table) == 1:
            print("total request handled: {}".format(main_memory.n_request))
            print("The overall execution time is: "+str(clk)+" cycles.")
            return clk
            #break
        clk += 1 #increment the clk counter


def start_run_new_hash(csr_ins):
    # setup
    csr_out = csr_ins*csr_ins
    matrix_space_bound = csr_ins.shape[0]*4 + csr_ins.nnz*8 
    N_ROWS = csr_ins.shape[0]
    main_memory = Memory(n_channels, matrix_space_bound, N_ROWS)
    test = get_nnzs(csr_ins, 7114)
    pe_array = []
    shared_complete_list = np.zeros(csr_out.shape[0])
    shared_status_table = np.zeros(n_pes)
    
    assigned_rows =[ *range(0, csr_ins.shape[0], 1) ]
    pe_array.append(newHashPE(0, n_pes, assigned_rows, csr_ins, csr_out, shared_complete_list, shared_status_table) )

    clk = 0    
    while(1):
        #gather returned request from all memory channels
        returned_requests = main_memory.tick()

        #process the received requests.
        for i in range(len(returned_requests)):
            curr_request = returned_requests[i]
            r_valid = curr_request[0]
            r_type = curr_request[1]
            r_id = curr_request[2]
            r_addr = curr_request[3]
            r_n = curr_request[4]
            # process valid and read request
            if(r_valid == 1 and r_type == 0): 
                pe_array[0].receive(curr_request)
        

        #gather the requests from pes and send it to main memory
        #for i in range(n_pes):
        curr_request_list = pe_array[0].tick()
        for i in range(len(curr_request_list)):
            curr_request = curr_request_list.pop(0)
            main_memory.enqueue(curr_request)
            pass

        if sum(shared_status_table) == 1:
            print("total request handled: {}".format(main_memory.n_request))
            print("The overall execution time is: "+str(clk)+" cycles.")
            return clk
            #break
        clk += 1 #increment the clk counter





if __name__ == "__main__":
    outfilename = "init_perf_comparision.log"
    stats_file = open(outfilename, "a+")
    csr_ins, filename = load_sparse_matrix(path_prefix)
    row_length_array = csr_ins.indptr[1:] - csr_ins.indptr[:-1]
    total_nnz = cal_all_nnzs(csr_ins, csr_ins)
    #print(total_nnz)
    # baseline_cycles = start_run(csr_ins)
    # print(baseline_cycles)
    #hash_cycles = start_run_hash(csr_ins)
    #print(hash_cycles)
    new_hash_cycles = start_run_new_hash(csr_ins)
    print(new_hash_cycles)
    #print("{} {} {} {} {} {} {} {}".format(filename, csr_ins.nnz, csr_ins.nnz/csr_ins.shape[0], \
    #    np.max(row_length_array), total_nnz, total_nnz/csr_ins.shape[0], baseline_cycles, new_hash_cycles), file=stats_file)


