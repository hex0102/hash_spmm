from utils import *


class HashCore:

    clk = 0
     
    def __init__(self, id, allocated_table_size, shared_hashtable, debug_counter, external_request):
        self.fetched_data = [-1, -1, 0, 0, 0] #valid, idx, val, freq, dirty
        self.id = id
        self.pe_mode = 0
        self.done = 0
        self.fetch_dest = 0 # 0: on-chip; 1: off-chip
        self.input_data = [0, -1, 0] # valid, idx, val
        self.allocated_table_size = allocated_table_size
        self.oc_probing_counter = 0
        self.shared_hashtable = shared_hashtable
        self.debug_counter = debug_counter
        self.write_complete = 1
        self.external_request = external_request
        

    def get_mode(self):
        return self.pe_mode
    
    def write_done(self, is_write_complete):
        self.write_complete = is_write_complete
    
    def get_write_status(self):
        return self.write_complete

    def receive(self, input_data = None, fetched_data = None):
        if input_data != None:
            self.input_data = input_data
        if fetched_data != None:
            self.fetched_data = fetched_data
    
    def set_external_request(self, external_request):
        self.external_request = external_request

    def set_table_size(self, on_chip_size, off_chip_size):
        self.allocated_table_size = on_chip_size
        self.offchip_table_size = off_chip_size

    def tick(self):
        nnz_processed_h = 0
        # output request info
        req_valid = 0
        req_type = 0
        req_addr = 0
        req_data = None #(0, 0, 0) # idx, val, freq

        # input data
        valid = self.input_data[0]
        col_idx = self.input_data[1]
        val = self.input_data[2]

        # returned_data
        fetch_valid = self.fetched_data[0] #if the block is fetched from the hash table
        fetch_idx = self.fetched_data[1]
        fetch_val = self.fetched_data[2]
        fetch_freq = self.fetched_data[3]
        fetch_dirty = self.fetched_data[4]

        if self.pe_mode == 1:
            if fetch_valid != -1:
                if fetch_idx == -1:
                    if fetch_dirty == 0:
                        fetch_idx = col_idx
                        fetch_val = val
                        req_valid = 1
                        req_type = 1
                        req_addr = self.hash_addr
                        req_data = [fetch_idx, fetch_val, fetch_freq]
                        self.done = 1
                        nnz_processed_h = 1
                        self.oc_probing_counter = 0
                        self.debug_counter[2].append([col_idx, self.hash_addr])
                    elif fetch_dirty == 1:
                        req_valid = 1
                        req_type = 0
                        req_addr = self.hash_addr
                        self.fetch_dest = 0    
                elif fetch_idx == col_idx:
                    fetch_val += val
                    req_valid = 1
                    req_type = 1
                    req_addr = self.hash_addr
                    req_data = [fetch_idx, fetch_val, fetch_freq]
                    self.done = 1
                    nnz_processed_h = 1
                    self.oc_probing_counter = 0
                    self.debug_counter[2].append([col_idx, self.hash_addr])
                else:
                    if self.oc_probing_counter < PROB_LENGTH:
                        self.hash_addr = (self.hash_addr + 1) & (self.allocated_table_size - 1)
                        req_valid = 1
                        req_type = 0
                        req_addr = self.hash_addr
                        self.oc_probing_counter += 1
                    else:
                        #print(col_idx)
                        self.pe_mode = 2 # change to mode 2 and access data off-chip
                        self.oc_probing_counter = 0
                self.fetched_data[0] = -1
                self.fetch_dest = 0    
        # process a new input
        elif self.pe_mode == 0 and self.write_complete == 1: # pe is in idle mode and accept new inputs and perform hash into OCM
            if valid != 0:
                self.pe_mode = 1 # compute hash
                self.hash_addr = hash_index(col_idx, 0, self.allocated_table_size)
                req_valid = 1
                req_type = 0
                req_addr = self.hash_addr
                self.input_data[0] = 0
                self.fetch_dest = 0
                self.write_complete = 0
                self.debug_counter[1].append(col_idx)
        elif self.pe_mode == 2: #
            self.pe_mode = 3
            self.hash_addr = hash_index(col_idx, 0, self.offchip_table_size)
            #print(col_idx)
            req_valid = 1
            req_type = 0
            req_addr = self.hash_addr
            self.input_data[0] = 0
            self.fetch_dest = 1
        elif self.pe_mode == 3:
            if fetch_valid != -1:
                if fetch_idx == -1:
                    fetch_idx = col_idx
                    fetch_val = val
                    req_valid = 1
                    req_type = 1
                    req_addr = self.hash_addr
                    req_data = [fetch_idx, fetch_val, fetch_freq]
                    self.done = 1
                    nnz_processed_h = 1
                    self.oc_probing_counter = 0
                    self.debug_counter[3].append(col_idx)
                    self.external_request[-1].append(self.hash_addr)
                elif fetch_idx == col_idx:
                    fetch_val += val
                    req_valid = 1
                    req_type = 1
                    req_addr = self.hash_addr
                    req_data = [fetch_idx, fetch_val, fetch_freq]
                    self.done = 1
                    nnz_processed_h = 1
                    self.oc_probing_counter = 0
                    self.debug_counter[3].append(col_idx)
                else:
                    self.hash_addr = (self.hash_addr + 1) & (self.offchip_table_size - 1)
                    req_valid = 1
                    req_type = 0
                    req_addr = self.hash_addr
                self.fetched_data[0] = -1
                self.fetch_dest = 1

        
        #idx 4B, val 4B, freq 4B
        if self.pe_mode == 2 or self.pe_mode == 3:
            returned_req = [self.fetch_dest, req_valid, req_type, PE_START_ID + self.id, req_addr*HASH_ITEM_SIZE, 1, req_data] 
        else:
            returned_req = [self.fetch_dest, self.id, req_valid, req_type, req_addr, req_data]  #6

        
        if self.done == 1:
            self.pe_mode = 0
            self.done = 0
            self.fetch_dest = 0
        self.clk += 1
        
        return returned_req, nnz_processed_h
