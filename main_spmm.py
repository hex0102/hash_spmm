import scipy
from scipy import io
import numpy as np
import sys

def select_row(csr_ins, row_id):
    return csr_ins.indices[csr_ins.indptr[row_id] : csr_ins.indptr[row_id + 1]]

#return the id of queues of least entreis
def select_least(data_queues):
    q_size = [ data_queues[i].size for i in range(n_queues)]
    queue_id = q_size.index(min(q_size))
    return queue_id



#this script compares on-chip read and write  of matraptor and hash-based method, assuming enough on-chip memory
if __name__ == "__main__":

    print_freq = 5000
    stats_file = open("stats.txt", "a+")
    path_prefix = "/home/xinhe/hash_spmm/"
    full_path = path_prefix + sys.argv[1]
    input_a = scipy.io.mmread(full_path) #"filter3D/filter3D.mtx"
    output_c = input_a*input_a
    
    data_name = full_path.rsplit('/', 1)[-1]
    print("A_NNZ:{} A_NNZ/ROW:{} C_NNZ:{} C_NNZ/ROW:{}".format(input_a.nnz, input_a.nnz/input_a.shape[0], output_c.nnz, output_c.nnz/output_c.shape[0]))

    n_row = input_a.shape[0]
    n_col = input_a.shape[1]
    assert(input_a.shape[0] == input_a.shape[1])
    input_a = input_a.tocsr()
    #for hash-based method: 
    print("hash-based method...")
    hash_r = 0
    hash_w = 0
    for i in range(n_row):
        if(i % print_freq == 0):    
            print("processing row_{}\n".format(i))        
        s_indptr = input_a.indptr[i]
        e_indptr = input_a.indptr[i+1]
        for j in range(s_indptr, e_indptr):
            list_of_id = select_row(input_a, input_a.indices[j])
            hash_r += list_of_id.size
            hash_w += list_of_id.size
    #for matraptor:
    n_queues = 10 # the actual queues is n_queues + 1
    matraptor_r = 0
    matraptor_w = 0
    matraptor_accu = 0
    print("counting matraptor...")
    for i in range(n_row):
        if(i % print_freq == 0):    
            print("processing row_{}\n".format(i))
        data_queues = [np.array([]) for _ in range(n_queues)]
        s_indptr = input_a.indptr[i]
        e_indptr = input_a.indptr[i+1]
        for j in range(s_indptr, e_indptr):
            smallest_queue_id = select_least(data_queues)
            selected_queue = data_queues[smallest_queue_id]
            list_of_id = select_row(input_a, input_a.indices[j])
            matraptor_r += selected_queue.size 
            merged_results = np.union1d(selected_queue, list_of_id)
            matraptor_accu += np.intersect1d(selected_queue, list_of_id).size
            data_queues[smallest_queue_id] = merged_results
            matraptor_w += merged_results.size
        #final merge for a row
        for w in range(n_queues):
            matraptor_r += data_queues[w].size
    print("Results:")        
    print(">>Hash-based\t read {} write {}".format(hash_r, hash_w))
    print(">>Matraptor\t read {} write {}".format(matraptor_r, matraptor_w))

    print("{},{},{},{},{},{},{},{},{},{}".format(data_name, input_a.shape[0], input_a.nnz, input_a.nnz/input_a.shape[0], output_c.nnz, output_c.nnz/output_c.shape[0], hash_r, hash_w, matraptor_r, matraptor_w), file=stats_file)
    print("Completed")

