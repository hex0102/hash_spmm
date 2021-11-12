import numpy as np
from utils import *
from hashsystem import *
DUMMY_ADDR = 0
FACTOR = 16
OUT_LIMIT = 32

A_DATA_BUFFER_SIZE = 64*FACTOR
A_PTR_BUFFER_SIZE = 8*FACTOR
B_DATA_BUFFER_SIZE = 256
B_PTR_BUFFER_SIZE = 32*FACTOR

CONCURRENT_C = 4

PRINT_FREQ = 100
MAXIMUM_SIZE = 131072
N_BANKS = 32

DEBUG = 0

class fullHash:
    request_buffer = []
    
    # n_loaders: number of fetchers/distributed buffers
    # assigned_row_ids: continous output idx
    # csr_m: input
    # csr_out: output
    # shared_complete_list: 
    # shared_status_table: 
    def __init__(self, n_loaders, n_pes, assigned_row_ids, csr_m, csr_out, shared_complete_list, shared_status_table):
        self.curr_data = None # current processed data from the unified FIFO
        self.shared_complete_list = shared_complete_list
        self.id = 0
        self.n_loaders = n_loaders
        self.n_pes = n_pes
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
        self.START_TEMP = self.START_C_INDICE_ADDR + 8 * csr_out.nnz
        
        
        self.overall_external_request = []
        self.external_request = [[ ]] # this records the address of external nonzeros for each output row
        self.write_start = []
        self.write_ptr = 0
        self.read_external_hashtable = []

        #data loaded by the fetchers/loaders
        self.stored_data = [[[], []], [[], []]]
        #stored the data in a FIFO
        self.unified_fifo = []
        #store the output in a multibank buffer/hash table
        self.hash_table = []
        #gather results into a buffer
        self.out_buffer = []
        self.ht_ptr = 0
        self.external_ht_ptr = 0

        

        # pointer
        self.have_ptr = []
        for i in range(self.n_loaders):
            self.stored_data[1][1].append([ ])
            self.have_ptr.append([-1, -1])  

        # pe_nnz
        self.pe_nnz = []
        for i in range(self.n_loaders):
            self.pe_nnz.append([ ])
            self.pe_nnz[i].append([0, 0, 0])

        self.pe_fiber_nnz = []
        for i in range(self.n_loaders):
            self.pe_fiber_nnz.append([ ])

        self.a_row_info = [] #this is used to track the start_idx and nnzs of a a_rows need to be processed
        self.n_b_row_ptr = 0

        self.fully_loaded = 0 #start storing
        self.processed = 0 #process all the nonzeros
        self.stored = 1 #stored all results
        
        self.count_nnz_per_outrow = 0
        self.nnz_outrow_list = []
        #self.nnz_outrow_list.append(0)

        self.all_nnz_counted = []

        self.cur_a_row_nnz = 0
        self.cur_b_row_nnz = []
        self.loader_done = []
        for i in range(n_loaders):
            self.cur_b_row_nnz.append(0)    
            self.loader_done.append(0)

        self.clk = 0
        self.selected_loader = 0
        self.selected_write_loader = 0
        self.n_assign_b = 0
        self.nnz_loaded = 0 # count the number of nnz loaded from the unified buffer
        self.nnz_processed = 0 # count the number of nnz processed by this PE for an output row
        self.a_ind_id = 0
        self.b_ptr_id = 0
        self.b_ind_id = 0
        self.cur_c_row_nnz = 0
        #self.write_dram_table_cc = 0


        self.request_q = []
        self.test = [] 

        
        self.external_hashtable = np.zeros((HASH_TABLE_OFF_CHIP, 4))#idx, val, freq 
        self.external_hashtable[:, 0] = -1  
        self.hashsystem = HashSystem(self.n_pes, MAXIMUM_SIZE, N_BANKS, self.unified_fifo, self.START_TEMP, self.external_hashtable, self.external_request)

        bound_nnz = hash_symbolic_upperbound(self.csr_m, self.csr_m)
        ht_size_array = next_power_of_2(bound_nnz)
        for i in range(ht_size_array.size):
            if(ht_size_array[i]>=0 and ht_size_array[i]<V_LEN):
                ht_size_array[i] = V_LEN  

        #for i in range(ht_size_array.size):
        #    ss = next_power_of_2_single(ht_size_array[i])

        ht_size_array = ht_size_array.astype(int)
        self.on_chip_ht_size = np.zeros(self.csr_m.shape[0]).astype(int)
        self.off_chip_ht_size = np.zeros(self.csr_m.shape[0]).astype(int)
        for i in range(self.csr_m.shape[0]):
            if ht_size_array[i] <= MAXIMUM_SIZE:
                self.on_chip_ht_size[i] = ht_size_array[i]
                self.off_chip_ht_size[i] = 0
            else:
                self.on_chip_ht_size[i] = MAXIMUM_SIZE
                self.off_chip_ht_size[i] = next_power_of_2_single(bound_nnz[i] - MAXIMUM_SIZE)

        self.track_row_id = 0 #track row id to allocate hash tables size    
        self.hashsystem.set_table_size(self.on_chip_ht_size[self.track_row_id], self.off_chip_ht_size[self.track_row_id])

    #when receving a block from main memory
    def receive(self, request):
        valid = request[0]
        type = request[1] # 0 for read, 1 for write
        # receive a valid read block
        if valid == 1 and type == 0 and request[2] < 999:
            addr = request[3]
            if (request[2]<PE_START_ID):
                self.n_outstanding -= 1
                # (load_a, load_data, col, val)
                # or (load_a, load_indptr, start_idx, nnz)
                received_data_tuple = select_data(request, self.csr_m)
                matrix_id = received_data_tuple[0]
                data_type = received_data_tuple[1]

                if matrix_id == 1 and data_type == 1:
                    find_and_fill(self.stored_data[matrix_id][data_type][request[2]], received_data_tuple, request)
                else:
                    find_and_fill(self.stored_data[matrix_id][data_type], received_data_tuple, request)
            elif (request[2]>=PE_START_ID and request[2]<PE_START_ID + self.n_pes): # a read quest from the hashcore
                self.n_outstanding -= 1
                received_data_tuple = select_external_data(request, self.external_hashtable, self.START_TEMP) #fast
                self.hashsystem.receive(received_data_tuple, request[2]  - PE_START_ID)
            elif request[2] == PE_START_ID + 2*self.n_pes: # a read request from final writer to read from the externel/dram hash table
                self.n_outstanding -= 1
                self.read_external_hashtable.append(request[3])
        
        if valid == 1 and type == 1 and request[2] >= PE_START_ID and request[2]<PE_START_ID + self.n_pes:
            # addr = request[3]
            # write_data_tuple = request[5]
            self.n_outstanding -= 1
            # table_idx =  (request[3] - self.START_TEMP)/HASH_ITEM_SIZE
            write_external_data(request, self.external_hashtable, self.START_TEMP)

        if valid == 1 and type == 0 and request[2] == 999:
            self.n_outstanding -= 1
            self.received_external_ht_request(request)
    
    #(1, 0, 999, self.START_TEMP + HASH_ITEM_SIZE*self.external_ht_ptr, local_n)
    def received_external_ht_request(self, external_request):
        # a valid read request for external ht was returned
        if external_request[0] == 1 and external_request[1] == 0:
            local_n = int(external_request[4])
            ext_ht_ptr = int((external_request[3] - self.START_TEMP)/HASH_ITEM_SIZE)
            selected_array = self.external_hashtable[ext_ht_ptr: ext_ht_ptr + local_n, :] 
            valid_idx = (selected_array[:, 0] != -1).nonzero()[0]
            if valid_idx.size != 0:
                valid_array = self.external_hashtable[valid_idx, 0:2]
                for i in range(valid_array.shape[0]):
                    curr_entry = [valid_array[i][0], valid_array[i][1]]
                    self.out_buffer.append(curr_entry)


    def update_pe_nnz(self, pe_id, nnz):
        self.pe_nnz[pe_id][-1][0] += nnz
        self.pe_nnz[pe_id][-1][1] = 1

    def pop_pe_nnz(self):
        for i in range(self.n_loaders):
            self.pe_nnz[i].pop(0)

    def set_pe_nnz(self):
        for i in range(self.n_loaders):
            self.pe_nnz[i][-1][1] = 1

    #when ticks, the PE 1) processes the elements and 2)sends r/w requests
    def tick(self):
        self.clk += 1

        #write the loaded data to a unified FIFO
        write_success = 0
        for i in range(self.n_loaders):
            # check self.selected_write_loader
            if self.pe_nnz[self.selected_write_loader][0][0] > 0 and len(self.stored_data[1][1][self.selected_write_loader]) > 0 and self.stored_data[1][1][self.selected_write_loader][0][0]!=-1:

                input_length =  min(len(self.stored_data[1][1][self.selected_write_loader]), self.n_pes, self.pe_fiber_nnz[self.selected_write_loader][0][0])
                for m in range(input_length):
                    if  self.stored_data[1][1][self.selected_write_loader][0][0] != -1 and self.pe_nnz[self.selected_write_loader][0][0]>0:
                        poped_b_nnz = self.stored_data[1][1][self.selected_write_loader][0]
                        self.unified_fifo.append([poped_b_nnz[0], poped_b_nnz[1]*self.pe_fiber_nnz[self.selected_write_loader][0][1], poped_b_nnz[2]])
                        self.stored_data[1][1][self.selected_write_loader].pop(0)
                        self.pe_nnz[self.selected_write_loader][0][0] -= 1
                        self.pe_fiber_nnz[self.selected_write_loader][0][0] -= 1
                    else:
                        break

                if self.pe_fiber_nnz[self.selected_write_loader][0][0] == 0:
                    self.pe_fiber_nnz[self.selected_write_loader].pop(0)    
                self.selected_write_loader += 1
                write_success = 1
            else:
                self.selected_write_loader += 1
            
            if self.selected_write_loader == self.n_loaders:
                self.selected_write_loader = 0
            if write_success:
                break
    
        #process elements from the unified FIFO
        if self.stored == 1:
            if len(self.nnz_outrow_list) and self.nnz_loaded < self.nnz_outrow_list[0] and len(self.unified_fifo)!=0:
                h_bound = min(self.n_pes, self.nnz_outrow_list[0] - self.nnz_loaded)
                n_loaded = self.hashsystem.send(h_bound)#self.hashsystem.tick()
                self.nnz_loaded += n_loaded
                #processed_data = self.unified_fifo[:self.n_pes]
                #del self.unified_fifo[:self.n_pes]
            nnz_process_in_cur_cycle, list_external_reqs = self.hashsystem.tick()
            

            for i in range(len(list_external_reqs)):
                #read/write from the hash 
                if list_external_reqs[i][0] == 1:
                    self.n_outstanding += 1
                self.request_q.append(list_external_reqs[i])

            self.nnz_processed += nnz_process_in_cur_cycle

            if  len(self.all_nnz_counted)!=0 and self.all_nnz_counted[0] == 1 and self.nnz_processed == self.nnz_outrow_list[0]:           
                    self.stored = 0
                    self.processed = 1
                    self.nnz_outrow_list.pop(0)
                    self.nnz_processed = 0
                    self.nnz_loaded = 0
                    #self.overall_external_request += self.external_request.pop(0)
                    #self.external_request.append([])
                    #self.hashsystem.set_external_request(self.external_request)
                    self.shared_complete_list[self.a_row_id[0]] = 1

   
            #if len(self.external_request) != 0: #put the addr of the valid external dram nnz per row to the overall buffer
            #    self.overall_external_request += self.external_request[0].copy()
            #    self.external_request.pop(0)

        curr_request = (0, 0, self.id, DUMMY_ADDR)
        
        # set a limit to the maximum number of outstanding requests a PE can have
        # configure it in the above declaration
        if self.n_outstanding < OUT_LIMIT:

            '''
            #squeeze the external ht entries and write back to DRAM
            if len(self.write_start) > 0:
                start_addr = self.write_start[0][0]
                total_nnz = self.write_start[0][1]
                nnz_leftover = total_nnz - self.write_ptr
                if nnz_leftover >= 8:
                    local_n = 8
                else:
                    local_n = nnz_leftover
                
                if len(self.read_external_hashtable) >= local_n:
                    if local_n > 0:
                        curr_request = (1, 1, PE_START_ID + 2*self.n_pes,  start_addr + self.write_ptr*HASH_ITEM_SIZE, local_n)
                        self.n_outstanding += 1
                        self.request_q.append(curr_request)
                    self.write_ptr += local_n
                    del self.read_external_hashtable[:local_n]
                    if self.write_ptr + local_n == total_nnz:
                        #finish writing external nnzs for this row
                        self.write_start.pop(0)
                        self.write_ptr = 0
                    
            # read dram HT request based on the hash core request
            if len(self.overall_external_request):
                cur_read_addr = self.overall_external_request.pop(0)
                curr_request = (1, 0, PE_START_ID + 2*self.n_pes,  self.START_TEMP + cur_read_addr*HASH_ITEM_SIZE, 1)
                self.n_outstanding += 1
                self.request_q.append(curr_request)
            '''    

            # write results back
            # only know where to write until PEs with smaller id finishes loading                               
            if self.processed and self.shared_complete_list[:self.a_row_id[0]].sum()==self.a_row_id[0]:
                curr_row_id = self.a_row_id[0]
                c_row_nnz = self.csr_out.indptr[curr_row_id+1] - self.csr_out.indptr[curr_row_id]

                local_n = 8
                
                #write the numbers in self.out_buffer to DRAM
                if len(self.out_buffer) >= 8:
                    curr_request = (1, 1, self.id,  int(self.START_C_INDICE_ADDR + self.csr_out.indptr[curr_row_id]*8) \
                        + self.cur_c_row_nnz*8, local_n)
                    self.cur_c_row_nnz += 8
                    #self.n_outstanding += 1
                    del self.out_buffer[:8]
                elif len(self.out_buffer) < 8 and self.ht_ptr >= self.hashsystem.on_chip_table_size:
                    curr_request = (1, 1, self.id,  int(self.START_C_INDICE_ADDR + self.csr_out.indptr[curr_row_id]*8) \
                        + self.cur_c_row_nnz*8, len(self.out_buffer))
                    self.cur_c_row_nnz += len(self.out_buffer)
                    #self.n_outstanding += 1
                    del self.out_buffer[:len(self.out_buffer)]
                else:
                    curr_request = (0, 1, self.id,  0, 8)
                
                self.request_q.append(curr_request)
                

                # the full hash table was scanned  and self.external_ht_ptr >= self.hashsystem.off_chip_table_size
                if self.ht_ptr >= self.hashsystem.on_chip_table_size and self.external_ht_ptr >= self.hashsystem.off_chip_table_size and len(self.out_buffer) == 0:
                    #self.write_start.append([int(self.START_C_INDICE_ADDR) + self.csr_out.indptr[curr_row_id]*8 + self.cur_c_row_nnz*8, c_row_nnz - self.cur_c_row_nnz])
                    self.a_row_id.pop(0)
                    self.fully_loaded = 0
                    self.processed = 0                        
                    self.cur_c_row_nnz = 0
                    self.ht_ptr = 0
                    self.external_ht_ptr = 0
                    self.stored = 1
                    self.pop_pe_nnz()
                    self.all_nnz_counted.pop(0)                
                    if self.rows_processed % PRINT_FREQ == 0:
                        print("PE"+str(self.id)+" finished "+str(self.rows_processed) +"th rows with "+ str(self.nnz_loaded) +" nnz processed.")
                    if(self.rows_processed == self.NUM_ASSIGNED_ROWS - 1):  
                        print(">>>Done: PE"+str(self.id)+" compleleted "+str(self.rows_processed) +" rows.")
                        self.shared_status_table[self.id] = 1
                        return None
                    self.rows_processed += 1
                    self.hashsystem.set_table_size(self.on_chip_ht_size[self.rows_processed], self.off_chip_ht_size[self.rows_processed])
                    self.hashsystem.reset_hashtable()                    
                
                
                # scan the external hashtable and push valid entries to out_buffer
                if len(self.out_buffer) < 128 and self.external_ht_ptr < self.hashsystem.off_chip_table_size:
                    print(self.hashsystem.off_chip_table_size)
                    if self.external_ht_ptr + 8 <= self.hashsystem.off_chip_table_size:
                        local_n = 8
                    else:
                        local_n = self.hashsystem.off_chip_table_size - self.external_ht_ptr

                    external_request = (1, 0, 999, self.START_TEMP + HASH_ITEM_SIZE*self.external_ht_ptr, local_n)
                    self.external_ht_ptr += local_n
                    self.n_outstanding += 1
                    self.request_q.append(external_request)
                

                # scan the hashtable and push valid entries to out_buffer
                if len(self.out_buffer) < 128 and self.ht_ptr < self.hashsystem.on_chip_table_size: #col_idx, val is 8B, so a block should have 8 pairs
                    n_available_slot = 128 - len(self.out_buffer)
                    if self.ht_ptr + 16 <= self.hashsystem.on_chip_table_size:
                        pop_length = 16  
                    else:
                        pop_length = self.hashsystem.on_chip_table_size - self.ht_ptr
                    pop_length = min(pop_length, n_available_slot)
                    pop_item = self.hashsystem.hashtable[self.ht_ptr : self.ht_ptr + pop_length, :]
                    pop_idx = pop_item[pop_item[:, 0]!= -1,:]
                    valid_item = pop_idx.tolist()
                    self.out_buffer = self.out_buffer + valid_item #append the items
                    self.ht_ptr += pop_length
                    #print(self.ht_ptr ) 
                # self.request_q.append(curr_request)
                
                '''
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
                        print("PE"+str(self.id)+" finished "+str(self.rows_processed) +"th rows with "+ str(self.nnz_loaded) +" nnz processed.")
                    if(self.rows_processed == self.NUM_ASSIGNED_ROWS - 1):  
                        print(">>>Done: PE"+str(self.id)+" compleleted "+str(self.rows_processed) +" rows.")
                        self.shared_status_table[self.id] = 1
                        return None
                    self.rows_processed += 1
                    #self.track_row_id = 0 #track row id to allocate hash tables size    
                    self.hashsystem.set_table_size(self.on_chip_ht_size[self.rows_processed], self.off_chip_ht_size[self.rows_processed])
                    #self.hashsystem.reset_hashtable()
                else:
                    c_valid = 1
                    curr_request = (c_valid, 1, self.id,  int(self.START_C_INDICE_ADDR + self.csr_out.indptr[curr_row_id]*8) \
                        + self.cur_c_row_nnz*8, local_n)    
                    self.cur_c_row_nnz += 8
                
                self.request_q.append(curr_request)
                #return curr_request
                '''




            # matrix a ptr into buffer stored_data[0][0] 
            # self.stored_data[0][0] (a_row_start_idx, a_row_nnz, request_addr), because request comes from main memory can be out-of-order
            if len(self.stored_data[0][0]) < A_PTR_BUFFER_SIZE:
                if(len(self.assigned_row_ids)):
                    a_row_id = self.assigned_row_ids.pop(0)
                    curr_request = (1, 0, self.id,  int(self.START_A_ADDR + 4*a_row_id), 1)
                    self.stored_data[0][0].append([-1, -1, int(self.START_A_ADDR + 4*a_row_id)])
                    self.n_outstanding += 1
                    self.request_q.append(curr_request)
                    #return curr_request    

            # load matrix a data into buffer stored_data[0][1]
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

            # matrix b ptr into buffer stored_data[1][0]: issue request to read the indptr of matrix b (row_start and nnz)
            if len(self.stored_data[1][0]) < B_PTR_BUFFER_SIZE:
                if len(self.stored_data[0][1]) != 0 and self.stored_data[0][1][0][0]!=-1:
                    curr_a_entry = self.stored_data[0][1].pop(0) # (col_id, val) of matrix a
                    curr_request = (1, 0, self.id,  int(self.START_B_ADDR + 4*curr_a_entry[0]), 1)
                    self.n_outstanding += 1
                    for m in range(1):
                        self.stored_data[1][0].append([-1, -1, int(self.START_B_ADDR + 4*curr_a_entry[0]), curr_a_entry[1]])
                    self.request_q.append(curr_request)


            # update the pe loader with b_ptr
            if len(self.stored_data[1][0]) and self.stored_data[1][0][0][0]!=-1:
                for m in range(self.n_loaders):
                    if len(self.stored_data[1][1][self.selected_loader]) < B_DATA_BUFFER_SIZE - 8 and self.have_ptr[self.selected_loader][0] == -1:
                        if self.a_row_info[0][1]!=0:
                            curr_entry = self.stored_data[1][0].pop(0)
                            self.have_ptr[self.selected_loader][0] = curr_entry[0] #b_start_indx
                            self.have_ptr[self.selected_loader][1] = curr_entry[1] #b_nnz
                            self.pe_fiber_nnz[self.selected_loader].append([curr_entry[1], curr_entry[3]]) 
                            self.count_nnz_per_outrow += curr_entry[1]

                            #b_row_start_idx = curr_entry[0]
                            #b_row_nnz = curr_entry[1]
                            self.update_pe_nnz(self.selected_loader, curr_entry[1])
                            self.n_assign_b += 1
                            if self.n_assign_b == self.a_row_info[0][1]:
                                self.n_assign_b = 0
                                self.a_row_info.pop(0)
                                self.set_pe_nnz()
                                self.nnz_outrow_list.append(self.count_nnz_per_outrow)
                                self.count_nnz_per_outrow = 0
                                self.all_nnz_counted.append(1)
                                for n in range(self.n_loaders):
                                    self.pe_nnz[n].append([0, 0, 0])

                            self.selected_loader += 1
                            if self.selected_loader == self.n_loaders:
                                self.selected_loader = 0                            
                            break
                        else:
                            self.set_pe_nnz()
                            self.a_row_info.pop(0)
                            self.nnz_outrow_list.append(0)
                            for n in range(self.n_loaders):
                                self.pe_nnz[n].append([0, 0, 0])
                            self.all_nnz_counted.append(1)
                            #break
                    else:
                        self.selected_loader += 1
                        if self.selected_loader == self.n_loaders:
                            self.selected_loader = 0
                        break
                    
            # loader send requests to main memory
            for i in range(self.n_loaders):
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
                                #self.count_nnz_per_outrow += self.have_ptr[i][1]
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