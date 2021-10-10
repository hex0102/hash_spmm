import numpy as np
from utils import *
DUMMY_ADDR = 0
FACTOR = 16
OUT_LIMIT = 8*FACTOR

A_DATA_BUFFER_SIZE = 64*FACTOR
A_PTR_BUFFER_SIZE = 8*FACTOR
B_DATA_BUFFER_SIZE = 256
B_PTR_BUFFER_SIZE = 32*FACTOR

CONCURRENT_C = 4

PRINT_FREQ = 1000

DEBUG = 0

class newHashPE:
    request_buffer = []
    
    def __init__(self, id, npes,assigned_row_ids, csr_m, csr_out, shared_complete_list, shared_status_table):
        self.curr_data = None
        self.shared_complete_list = shared_complete_list
        self.id = id
        self.npes = npes
        self.shared_status_table = shared_status_table
        self.assigned_row_ids = assigned_row_ids.copy() # output rows processed by this PE

        self.NUM_ASSIGNED_ROWS = len(assigned_row_ids) #16000 #len(assigned_row_ids)
        self.rows_processed = 0
        self.leftover = -1

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

        self.have_ptr = []
        for i in range(self.npes):
            self.stored_data[1][1].append([ ])
            self.have_ptr.append([-1, -1])  

        self.pe_nnz = []
        for i in range(self.npes):
            self.pe_nnz.append([])
            self.pe_nnz[i].append([0, 0])

        self.a_row_info = [] #this is used to track the start_idx and nnzs of a a_rows need to be processed
        self.n_b_row_ptr = 0

        # = [] #this is used to track the assigned row_id 
        self.fully_loaded = 0 #start storing
        self.processed = 0 #process all the nonzeros
        self.stored = 1 #stored all results
        
        self.count_nnz_per_outrow = 0
        self.nnz_outrow_list = []

        self.all_nnz_counted = []

        self.cur_a_row_nnz = 0
        self.cur_b_row_nnz = []
        self.pe_done = []
        for i in range(npes):
            self.cur_b_row_nnz.append(0)    
            self.pe_done.append(0)

        self.clk = 0
        self.selected_pe = 0
        self.n_assign_b = 0
        self.nnz_processed = 0 # count the number of nnz processed by this PE for an output row


        self.a_ptr_id = 0
        self.a_ind_id = 0
        self.b_ptr_id = 0
        self.b_ind_id = 0
        self.cur_c_row_nnz = 0

        self.request_q = []
        self.test = []       

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
            if matrix_id == 1 and data_type == 1:
                find_and_fill(self.stored_data[matrix_id][data_type][request[2]], received_data_tuple, request)
            else:
                find_and_fill(self.stored_data[matrix_id][data_type], received_data_tuple, request)

    def update_pe_nnz(self, pe_id, nnz):
        self.pe_nnz[pe_id][-1][0] += nnz
        self.pe_nnz[pe_id][-1][1] = 1

    def pop_pe_nnz(self):
        for i in range(self.npes):
            self.pe_nnz[i].pop(0)

    def set_pe_nnz(self):
        for i in range(self.npes):
            #if self.pe_nnz[i][-1] == 0:
            self.pe_nnz[i][-1][1] = 1

    #when ticks, the PE 1) processes the elements and 2)sends r/w requests
    def tick(self):
        self.clk += 1
        # process elements from rows of matrix B
        pe_processed = 0
        if self.stored == 1:
            i = 0
            #for i in range(self.npes):
            while check_valid(self.stored_data[1][1], self.pe_done) and pe_processed < self.npes:
                if( len(self.stored_data[1][1][i]) > 0 and self.stored_data[1][1][i][0][0]!=-1):
                    if len(self.pe_nnz[i])!=0 and self.pe_nnz[i][0][0]!=0 and self.pe_nnz[i][0][1] == 1:                 
                        '''
                        if DEBUG == 1:
                            if self.rows_processed == 7114:
                                self.test.append(self.curr_data[0])
                        '''
                        self.curr_data = self.stored_data[1][1][i].pop(0)
                        self.nnz_processed += 1
                        self.pe_nnz[i][0][0] -= 1

                    if len(self.pe_nnz[i])!=0 and self.pe_nnz[i][0][1] == 1 and len(self.all_nnz_counted)!=0 and self.pe_nnz[i][0][0] == 0 and self.all_nnz_counted[0] == 1:
                        self.pe_done[i] = 1
                
                # remaining pe nnz for a row  AND remaing nnz is 0 and
                if len(self.pe_nnz[i])!=0 and self.pe_nnz[i][0][0]==0 and self.pe_nnz[i][0][1] == 1 and len(self.all_nnz_counted)!=0 and self.all_nnz_counted[0] == 1:
                    self.pe_done[i] = 1
                i += 1
                if(i == self.npes):
                    i = 0


        if  sum(self.pe_done) == self.npes: #self.nnz_processed == self.nnz_outrow_list[0] or self.nnz_outrow_list[0]==0:
            self.stored = 0
            self.processed = 1
            self.shared_complete_list[self.a_row_id[0]] = 1
            for m in range(self.npes):
                self.pe_done[m] = 0
            if DEBUG == 1:
                if self.rows_processed == 7114:
                    self.test.sort()
                    init_array = get_nnzs(self.csr_m, 7114)
                    init_array.sort()
                    for m in range(len(self.test)):
                        print(init_array[m])
                        print(self.test[m])
                        if(init_array[m] != self.test[m]):
                            print("error")

            

        curr_request = (0, 0, self.id, DUMMY_ADDR)
        # set a limit to the maximum number of outstanding requests a PE can have
        # configure it in the above declaration
        if self.n_outstanding < OUT_LIMIT:


            # write results back
            # only know where to write until PEs with smaller id finishes loading
                                
            if self.processed and self.shared_complete_list[:self.a_row_id[0]].sum()==self.a_row_id[0]:
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
                    self.pop_pe_nnz()
                    self.all_nnz_counted.pop(0)
                    if self.rows_processed % PRINT_FREQ == 0:
                        print("PE"+str(self.id)+" finished "+str(self.rows_processed) +"th rows with "+ str(self.nnz_processed) +" nnz processed.")
                    if(self.rows_processed == self.NUM_ASSIGNED_ROWS - 1):  
                        print(">>>Done: PE"+str(self.id)+" compleleted "+str(self.rows_processed) +" rows.")
                        self.shared_status_table[self.id] = 1
                    self.rows_processed += 1
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
                        self.a_row_info.append(self.stored_data[0][0][0]) # holds the start_id and nnz of the curr_a_row

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


            # update the pe loader with b_ptr
            if len(self.stored_data[1][0]) and self.stored_data[1][0][0][0]!=-1:
                for m in range(self.npes):
                    if len(self.stored_data[1][1][self.selected_pe]) < B_DATA_BUFFER_SIZE - 8 and self.have_ptr[self.selected_pe][0] == -1:
                        if self.a_row_info[0][1]!=0:
                            curr_entry = self.stored_data[1][0].pop(0)
                            self.have_ptr[self.selected_pe][0] = curr_entry[0]
                            self.have_ptr[self.selected_pe][1] = curr_entry[1]         
                            #b_row_start_idx = curr_entry[0]
                            #b_row_nnz = curr_entry[1]
                            self.update_pe_nnz(self.selected_pe, curr_entry[1])
                            self.n_assign_b += 1
                            if self.n_assign_b == self.a_row_info[0][1]:
                                self.n_assign_b = 0
                                self.a_row_info.pop(0)
                                self.set_pe_nnz()
                                self.all_nnz_counted.append(1)
                                for n in range(self.npes):
                                    self.pe_nnz[n].append([0, 0])

                            self.selected_pe += 1
                            if self.selected_pe == self.npes:
                                self.selected_pe = 0                            
                            break
                        else:
                            self.set_pe_nnz()
                            self.a_row_info.pop(0)
                            for n in range(self.npes):
                                self.pe_nnz[n].append([0, 0])
                            self.all_nnz_counted.append(1)
                            #break
                    else:
                        self.selected_pe += 1
                        if self.selected_pe == self.npes:
                            self.selected_pe = 0
                        break
                    
            # pe loader send requests to main memory
            for i in range(self.npes):
                if len(self.stored_data[1][1][i]) < B_DATA_BUFFER_SIZE - 8:
                    if self.have_ptr[i][0] != -1:
                        b_row_start_idx = self.have_ptr[i][0]
                        b_row_nnz = self.have_ptr[i][1]
                        
                        exit_s = 0
                        for mm in range(CONCURRENT_C):
                            local_n = 8
                            addr_n = int(self.START_B_INDICE_ADDR + b_row_start_idx*8) + self.cur_b_row_nnz[i]*8
                            if self.cur_b_row_nnz[i] +8 >= b_row_nnz:
                                #curr_entry = self.stored_data[1][0].pop(0)

                                local_n = 8 - (self.cur_b_row_nnz[i] + 8 - b_row_nnz)
                                if b_row_nnz != 0:
                                    c_valid = 1
                                else:
                                    c_valid = 0
                                curr_request = (c_valid, 0, i,  int(self.START_B_INDICE_ADDR + b_row_start_idx*8) \
                                    + self.cur_b_row_nnz[i]*8, local_n)  
                                self.cur_b_row_nnz[i] = 0

                                self.n_b_row_ptr += 1 #whenever a row_ptr of B is read
                                self.count_nnz_per_outrow += self.have_ptr[i][1]
                                self.have_ptr[i][0] = -1
                                self.have_ptr[i][1] = -1
                                exit_s = 1
                            else:
                                c_valid = 1
                                curr_request = (c_valid, 0, i,  int(self.START_B_INDICE_ADDR + b_row_start_idx*8) \
                                    + self.cur_b_row_nnz[i]*8, local_n)  
                                self.cur_b_row_nnz[i] += 8
                            if c_valid == 1:
                                for m in range(local_n):
                                    self.stored_data[1][1][i].append([-1, -1, addr_n])
                                self.n_outstanding += 1
                            
                            self.request_q.append(curr_request)
                            if exit_s:
                                break

        return self.request_q