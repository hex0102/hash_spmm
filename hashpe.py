import numpy as np
from utils import *
DUMMY_ADDR = 0
FACTOR = 16
OUT_LIMIT = 8*FACTOR
B_DATA_BUFFER_SIZE = 256*FACTOR
B_PTR_BUFFER_SIZE = 32*FACTOR
A_DATA_BUFFER_SIZE = 64*FACTOR
A_PTR_BUFFER_SIZE = 8*FACTOR
CONCURRENT_C = 4

PRINT_FREQ = 500

class HashPE:
    request_buffer = []
    
    def __init__(self, id, vector_count,assigned_row_ids, csr_m, csr_out, shared_complete_list, shared_status_table):
        self.curr_data = None
        self.shared_complete_list = shared_complete_list
        self.id = id
        self.vector_count = vector_count
        self.shared_status_table = shared_status_table
        self.assigned_row_ids = assigned_row_ids.copy() # output rows processed by this PE

        self.NUM_ASSIGNED_ROWS = len(assigned_row_ids)
        self.rows_processed = 0

        self.a_row_id = assigned_row_ids.copy()
        self.a_colidx = [] # load the corresponding column of matrix A    
        self.n_outstanding = 0
        self.csr_m = csr_m # input matrix
        self.csr_out = csr_out # output matrix

        self.cur_row_nnz = 0 # NNZs in the curr row of B
        self.already_read = 0 # NNZs already read in the curr row of B

        self.START_A_ADDR = 0
        self.START_A_INDICE_ADDR = 4 * csr_m.shape[0]
        self.START_B_ADDR = 4 * csr_m.shape[0] + 8 * csr_m.nnz
        self.START_B_INDICE_ADDR = self.START_B_ADDR + 4 * csr_m.shape[0]
        self.START_C_ADDR = 2 * self.START_B_ADDR 
        self.START_C_INDICE_ADDR = self.START_C_ADDR + 4 * csr_out.shape[0]

        self.stored_data = [[[], []], [[], []]]
        self.a_row_info = [] #this is used to track the start_idx and nnzs of a a_rows need to be processed
        self.n_b_row_ptr = 0

        # = [] #this is used to track the assigned row_id 
        self.fully_loaded = 0 #start storing
        self.processed = 0 #process all the nonzeros
        self.stored = 1 #stored all results
        
        self.count_nnz_per_outrow = 0
        self.nnz_outrow_list = []

        self.cur_a_row_nnz = 0
        self.cur_b_row_nnz = 0

        self.clk = 0

        self.nnz_processed = 0 # count the number of nnz processed by this PE for an output row

        self.a_ptr_id = 0
        self.a_ind_id = 0
        self.b_ptr_id = 0
        self.b_ind_id = 0
        self.cur_c_row_nnz = 0

        self.request_q = []       

    def receive(self, request):
        valid = request[0]
        type = request[1] # 0 for read, 1 for write
        if valid == 1 and type == 0:
            addr = request[3]
            self.n_outstanding -= 1
            # (load_a, load_data, col, val)
            # or (load_a, load_indptr, start_idx, nnz)
            received_data_tuple = select_data(request, self.csr_m)
            matrix_id = received_data_tuple[0]
            data_type = received_data_tuple[1]
            #self.stored_data[matrix_id][data_type].extend(received_data_tuple[2])
            find_and_fill(self.stored_data[matrix_id][data_type], received_data_tuple)
            
    #when ticks, the PE 1) processes the elements and 2)sends r/w requests
    def tick(self):
        self.clk+=1
        # process elements from rows of matrix B
        
        if(len(self.stored_data[1][1]) > 0 and self.stored_data[1][1][0][0]!=-1 ):
            for i in range(len(self.stored_data[1][1][:self.vector_count])):
                if self.stored_data[1][1][0][0] != -1:
                    self.curr_data = self.stored_data[1][1].pop(0)
                    self.nnz_processed += 1
                else:
                    break
        
        if(len(self.nnz_outrow_list)):
            if (self.nnz_processed == self.nnz_outrow_list[0] or self.nnz_outrow_list[0]==0) and self.fully_loaded:
                self.processed = 1
                self.shared_complete_list[self.a_row_id[0]] = 1
                self.nnz_outrow_list.pop(0)
                self.nnz_processed = 0
            

        curr_request = (0, 0, self.id, DUMMY_ADDR)
        # set a limit to the maximum number of outstanding requests a PE can have
        # configure it in the above declaration
        if self.n_outstanding < OUT_LIMIT:


            # write results back
            # only know where to write until PEs with smaller id finishes loading
                                
            if self.fully_loaded and self.processed and self.shared_complete_list[:self.a_row_id[0]].sum()==self.a_row_id[0]:
                curr_row_id = self.a_row_id[0]
                c_row_nnz = self.csr_out.indptr[curr_row_id+1] - self.csr_out.indptr[curr_row_id]
                #if self.csr_out.indptr:
                local_n = 8
                
                if self.cur_c_row_nnz + 8 >= c_row_nnz: 
                    local_n = 8 - (self.cur_c_row_nnz + 8 - c_row_nnz)
                    self.a_row_id.pop(0)
                    self.fully_loaded = 0
                    self.processed = 0
                    if c_row_nnz != 0:
                        c_valid = 1
                    else:
                        c_valid = 0  
                    curr_request = (c_valid, 1, self.id,  int(self.START_C_INDICE_ADDR + self.csr_out.indptr[curr_row_id]*8) \
                        + self.cur_c_row_nnz*8, local_n)                       
                    self.cur_c_row_nnz = 0
                    self.stored = 1
                    self.rows_processed += 1
                    if self.rows_processed % PRINT_FREQ == 0:
                        print("PE"+str(self.id)+" finished "+str(self.rows_processed) +"th rows.")
                    if(self.rows_processed == self.NUM_ASSIGNED_ROWS):  
                        print(">>>Done: PE"+str(self.id)+" compleleted "+str(self.rows_processed) +" rows.")
                        self.shared_status_table[self.id] = 1
                else:
                    c_valid = 1
                    curr_request = (c_valid, 1, self.id,  int(self.START_C_INDICE_ADDR + self.csr_out.indptr[curr_row_id]*8) \
                        + self.cur_c_row_nnz*8, local_n)    
                    self.cur_c_row_nnz += 8
                
                self.request_q.append(curr_request)
                #return curr_request




            # matrix a and ptr buffer
            if len(self.stored_data[0][0]) < A_PTR_BUFFER_SIZE:
                if(len(self.assigned_row_ids)):
                    a_row_id = self.assigned_row_ids.pop(0)
                    curr_request = (1, 0, self.id,  int(self.START_A_ADDR + 4*a_row_id), 1)
                    self.stored_data[0][0].append([-1, -1, int(self.START_A_ADDR + 4*a_row_id)])
                    self.a_ptr_id += 1
                    self.n_outstanding += 1
                    self.request_q.append(curr_request)
                    #return curr_request    

            # matrix a and data buffer
            if len(self.stored_data[0][1]) < A_DATA_BUFFER_SIZE - 8:
                if len(self.stored_data[0][0]) and self.stored_data[0][0][0][0]!=-1:
                    a_row_start_idx = self.stored_data[0][0][0][0]
                    a_row_nnz = self.stored_data[0][0][0][1]       
                    local_n = 8 
                    addr_n = int(self.START_A_INDICE_ADDR + a_row_start_idx*8) + self.cur_a_row_nnz*8
                    if(self.cur_a_row_nnz == 0):
                        self.a_row_info.append(self.stored_data[0][0][0])

                    if self.cur_a_row_nnz + 8 >= a_row_nnz:
                        curr_ptr = self.stored_data[0][0].pop(0)
                        local_n = 8 - (self.cur_a_row_nnz + 8 - a_row_nnz)
                        if a_row_nnz != 0:
                            c_valid = 1
                        else:
                            c_valid = 0                        
                        curr_request = (c_valid, 0, self.id,  int(self.START_A_INDICE_ADDR + a_row_start_idx*8) \
                            + self.cur_a_row_nnz*8, local_n)
                        self.cur_a_row_nnz = 0

                        
                        #self.a_row_info.append(curr_ptr) #hold the number of rows needed for each output row (in-order)
                    
                    else:
                        c_valid = 1
                        curr_request = (1, 0, self.id,  int(self.START_A_INDICE_ADDR + a_row_start_idx*8) \
                            + self.cur_a_row_nnz*8, local_n)
                        self.cur_a_row_nnz += 8 # 64B has 8 (col_idx, val) tuples    
                    
                    if c_valid == 1:
                        for m in range(local_n):
                            self.stored_data[0][1].append([-1, -1, addr_n])
                    
                    if c_valid == 1:
                        self.n_outstanding += 1
                    self.request_q.append(curr_request)
                    #return curr_request

            # matrix b and ptr buffer: issue request to read the indptr of matrix b (row_start and nnz)
            if len(self.stored_data[1][0]) < B_PTR_BUFFER_SIZE:
                # matrix_a and data buffer
                if len(self.stored_data[0][1]) != 0 and self.stored_data[0][1][0][0]!=-1:
                    curr_a_entry = self.stored_data[0][1].pop(0) # (col_id, val) of matrix a
                    #if curr_a_entry
                    curr_request = (1, 0, self.id,  int(self.START_B_ADDR + 4*curr_a_entry[0]), 1)
                    self.n_outstanding += 1
                    for m in range(1):
                        self.stored_data[1][0].append([-1, -1, int(self.START_B_ADDR + 4*curr_a_entry[0])])
                    self.request_q.append(curr_request)
                    #return curr_request 


            # matrix b and data buffer
            if len(self.stored_data[1][1]) < B_DATA_BUFFER_SIZE - 8 and self.stored == 1:
                # todo: check if all nnz of this row is fetched
                if len(self.stored_data[1][0]) and self.stored_data[1][0][0][0]!=-1 :
                    if self.a_row_info[0][1]!=0:
                        b_row_start_idx = self.stored_data[1][0][0][0]
                        b_row_nnz = self.stored_data[1][0][0][1]
                        b_addr = self.stored_data[1][0][0][2]
                        exit_s = 0
                        #if not (b_row_start_idx == -1 and b_row_nnz == 0 and b_addr == -1):
                        for mm in range(CONCURRENT_C):
                            local_n = 8
                            addr_n = int(self.START_B_INDICE_ADDR + b_row_start_idx*8) + self.cur_b_row_nnz*8
                            if self.cur_b_row_nnz +8 >= b_row_nnz:
                                curr_entry = self.stored_data[1][0].pop(0)
                                local_n = 8 - (self.cur_b_row_nnz + 8 - b_row_nnz)
                                if b_row_nnz != 0:
                                    c_valid = 1
                                else:
                                    c_valid = 0
                                curr_request = (c_valid, 0, self.id,  int(self.START_B_INDICE_ADDR + b_row_start_idx*8) \
                                    + self.cur_b_row_nnz*8, local_n)  
                                self.cur_b_row_nnz = 0

                                self.n_b_row_ptr += 1 #whenever a row_ptr of B is read
                                self.count_nnz_per_outrow += curr_entry[1]
                                exit_s = 1

                            else:
                                c_valid = 1
                                curr_request = (c_valid, 0, self.id,  int(self.START_B_INDICE_ADDR + b_row_start_idx*8) \
                                    + self.cur_b_row_nnz*8, local_n)  
                                self.cur_b_row_nnz += 8
                            if c_valid == 1:
                                for m in range(local_n):
                                    self.stored_data[1][1].append([-1, -1, addr_n])
                                self.n_outstanding += 1
                            
                            self.request_q.append(curr_request)
                            if exit_s:
                                break

                if len(self.a_row_info):   #self.a_row_info[0][1]==25 and self.rows_processed > 115                 
                    if self.n_b_row_ptr == self.a_row_info[0][1]:
                        self.n_b_row_ptr = 0
                        self.a_row_info.pop(0)
                        self.fully_loaded = 1
                        self.nnz_outrow_list.append(self.count_nnz_per_outrow)
                        self.count_nnz_per_outrow = 0
                        self.stored = 0

                    #return curr_request

                



            
        




        return self.request_q