import numpy as np
import scipy
from scipy import io
import sys
import random
from random import randint


def hash_index(key, i, table_size):
    if i == 0:
        hash_addr = (key * 107) & (table_size - 1)
    return hash_addr

def get_source(addr, matrix_space_bound, N_ROWS):
    source_name = ''
    if addr < matrix_space_bound:
        source_name += "load_a"
    else:
        addr -= matrix_space_bound
        source_name += "load_b"
    
    if addr < N_ROWS*4:
        source_name += "_ptr"
    else:
        source_name += "_data"
    
    return source_name

def select_data(request, csr_ins):
    addr = request[3]
    local_n = request[4]
    matrix_space_bound = csr_ins.shape[0]*4 + csr_ins.nnz*8 
    if addr < matrix_space_bound:
        load_a = 0
    else:
        addr -= matrix_space_bound
        load_a = 1
    N_ROWS = csr_ins.shape[0]
    if addr < N_ROWS*4:
        load_ptr = 0
        first_v = [csr_ins.indptr[int(addr/4)]]
        second_v = [csr_ins.indptr[int(addr/4)+1] - csr_ins.indptr[int(addr/4)]]
        combined_v = np.column_stack((first_v, second_v)).tolist()
    else:
        #load a data block (8 pairs ßat most)
        load_ptr = 1
        idx = int((addr/4 - N_ROWS)/2)
        first_v = csr_ins.indices[ idx : idx + local_n]
        second_v = csr_ins.data[ idx : idx + local_n]
        combined_v = np.column_stack((first_v, second_v)).tolist()
    return (load_a, load_ptr, combined_v, request[3])

def find_and_fill(store_list, loaded_blk, request=None):
    addr_n = loaded_blk[3]
    total_length = len(loaded_blk[2])

    store_np = np.array(store_list)
    #it could be that request with same addr exists, so fill the early requestíí
    return_indices = np.where( (store_np[:, 2]==addr_n) & (store_np[:, 0]==-1))
    first_index = return_indices[0][0]
    #temp = (np.array(store_list)[:, 2]).tolist()
    #first_index = temp.index(addr_n)
    for i in range(total_length):
        store_list[first_index + i][0] = loaded_blk[2][i][0]
        store_list[first_index + i][1] = loaded_blk[2][i][1]

def cal_nnz(csr_ins, row_id):
    row_id_array = select_row_ids(csr_ins, row_id)
    cc = 0
    for i in range(row_id_array.size):
        cc += select_row_ids(csr_ins, row_id_array[i]).size
        #print(select_row_ids(csr_ins, row_id_array[i]).size)
    return cc

def getNNZ_by_id(csr_a, csr_b, row_id):
    row_id_array = select_row_ids(csr_a, row_id)
    result_np_array = np.array([]).astype(int)
    for i in range(row_id_array.size):
        result_np_array = np.concatenate([result_np_array, select_row_ids(csr_b, row_id_array[i])])
    return result_np_array 


def get_nnzs(csr_ins, row_id):
    row_id_array = select_row_ids(csr_ins, row_id)
    result_np_array = np.array([])
    for i in range(row_id_array.size):
        result_np_array = np.concatenate([result_np_array, select_row_ids(csr_ins, row_id_array[i])])  
    return result_np_array       

def cal_all_nnzs(csr_a, csr_b):
    cc = csr_a.nnz
    for i in range(csr_a.shape[0]):
        a_row_ids = select_row_ids(csr_a, i)
        for j in range(a_row_ids.size):
            cc += select_row_ids(csr_b, a_row_ids[j]).size
    return cc


def check_valid(input_data, pe_done):
    for i in range(len(input_data)):
        if pe_done[i] == 1:
            continue
        if len(input_data[i]) > 0 and input_data[i][0][0]!=-1:
            return True

def cal_nnzs_idx(csr_a, csr_b, idx):
    cc = 0
    for i in range(idx):
        a_row_ids = select_row_ids(csr_a, i)
        for j in range(a_row_ids.size):
            cc += select_row_ids(csr_b, a_row_ids[j]).size
    return cc


def select_row_ids(csr_ins, row_id):
    return csr_ins.indices[csr_ins.indptr[row_id] : csr_ins.indptr[row_id + 1]]

def select_row_data(csr_ins, row_id):
    return csr_ins.data[csr_ins.indptr[row_id] : csr_ins.indptr[row_id + 1]]

def get_row_lengths(csr_ins, row_ids):
    partial_csr = csr_ins[row_ids, :]
    return partial_csr.indptr[1:] - partial_csr.indptr[:-1]

def fifo_read(buf, buffer_head, buffer_tail, fifo_size):
    valid = 0
    val = None
    if buffer_head != buffer_tail: #if theres avaiable entry
        valid = 1
        val = buf[buffer_tail]
        #buffer_tail += 1
        #if buffer_tail == fifo_size:
        #    buffer_tail = 0
    else:
        valid = 0

    return valid, val, buffer_head, buffer_tail

def fifo_pop(buf, buffer_head, buffer_tail, fifo_size):
    buffer_tail += 1
    if buffer_tail == fifo_size:
        buffer_tail = 0


def fifo_write(buf, buffer_head, buffer_tail, fifo_size, gap):
    valid = 1
    if (buffer_head + gap >= buffer_tail and buffer_tail > buffer_head ) or (  buffer_head + gap >= fifo_size and buffer_tail <= buffer_head + gap - fifo_size) :
        #no more room
        valid = 0
    return valid


def load_sparse_matrix(path_prefix):
    if len(sys.argv) == 4:
        if int(sys.argv[1]) == 0:
            # 0 262144 0.00002
            # 0 20000 0.0002048
            n_nodes = int(sys.argv[2])  # 65536*4
            density = float(sys.argv[3]) #0.00002
            random.seed(10)
            NNZ = int(n_nodes * n_nodes * density)
            points = {(randint(0, n_nodes * n_nodes - 1)) for i in range(NNZ)}
            while len(points) < NNZ:
                points |= {(randint(0, n_nodes * n_nodes - 1))}
            all_ids = np.array(list(x for x in points))
            all_ids = np.sort(all_ids)
            row_ids = all_ids // n_nodes
            col_ids = all_ids % n_nodes
            filename = "test_"+str(n_nodes)+"_"+str(density)+".data"
        else:
            #1 Slashdot0902.txt 948464
            #1 facebook_combined.txt 88234
            #1 soc-Epinions1.txt 508837
            #1 Wiki-Vote.txt 103689
            #1 com-amazon.ungraph.txt 925872
            #1 com-youtube.ungraph.txt 2987624
            #1 com-dblp.ungraph.txt 1049866
            #1 flickrEdges.txt 2316948
            #1 roadNet-CA.txt 5533214
            #1 Email-EuAll.txt 420045

            ##1 twitter_combined.txt 2420766  buggy
            

            #1 RMAT_SY1024_0.16.data   167772
            #1 RMAT_SY2048_0.04.data  167772
            #1 RMAT_SY4096_0.01.data  167772
            #1 RMAT_SY8192_0.0025.data  167772
            #1 RMAT_SY16384_0.000625.data 167772
            #1 RMAT_SY32768_0.00015625.data 167772

            #1 RMAT_SY16384_0.0003125.data  
            #1 RMAT_SY16384_0.000625.data  
            #1 RMAT_SY16384_0.00125.data  
            #1 RMAT_SY16384_0.0025.data  
            #1 RMAT_SY16384_0.005.data  
            #1 RMAT_SY16384_0.01.data  

            filename = sys.argv[2]
            n_edge = int(sys.argv[3])

            coo_list_t = np.zeros((n_edge, 2))
            with open(filename) as f:
                curr_line = f.readline()
                cc = 0
                while (curr_line):
                    if "#" in curr_line:
                        curr_line = f.readline()
                        continue
                    list_num = curr_line.split()
                    coo_list_t[cc, 0] = int(list_num[0])
                    coo_list_t[cc, 1] = int(list_num[1])
                    cc += 1
                    curr_line = f.readline()
            row_ids = coo_list_t[:, 1].astype(int)
            col_ids = coo_list_t[:, 0].astype(int)
            output_filename = "inputs/"+filename[0:-4]+ "_"+str(n_edge) + '.data'
    elif len(sys.argv) == 2:
        #path_prefix += "inputs/"
        full_path = "inputs/" + sys.argv[1] #/home/xinhe/flextpu-micro/af23560.mtx
        input_a = scipy.io.mmread(full_path) #"filter3D/filter3D.mtx" 
        data_name = full_path.rsplit('/', 1)[-1]
        input_a = input_a.tocoo()
        row_ids = input_a.row
        col_ids = input_a.col
        filename = sys.argv[1]
    else:
        print("Incorrect number of argv...")


    # print("Loaded/Generated matrix data...")
    NNZ = row_ids.size
    current_idx_slices = np.concatenate([row_ids, col_ids])
    # number of unique id (the RANK of the matrix)
    n_unique_col = np.unique(current_idx_slices).size
    indx_range = range(n_unique_col)
    sorted_cols = np.sort(np.unique(current_idx_slices)).tolist()
    mapping_dict = dict(zip(sorted_cols, indx_range))
    #remap the index so that isolated nodes are removed 
    current_idx_slices = np.asarray(list(map(np.vectorize(mapping_dict.get), list(np.stack([row_ids, col_ids])))))

    #val_array = np.arange(0, current_idx_slices[1, :].size)*0.0001 + 0.0001
    val_array = (current_idx_slices[1, :] + current_idx_slices[0, :])*0.0001 + 0.0001

    uniform_coo = scipy.sparse.coo_matrix(
        (val_array, (current_idx_slices[1, :], current_idx_slices[0, :])),
        shape=(n_unique_col, n_unique_col))
    uniform_csr = uniform_coo.tocsr()
    
    print("Loaded/Generated matrix data...")

    return uniform_csr, filename.rsplit('.', 1)[0]