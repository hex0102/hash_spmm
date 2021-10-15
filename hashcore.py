from utils import *

class HashCore:

    clk = 0
    fetched_data = [0, -1, 0, 0] #valid, idx, val, freq
    def __init__(self, id, allocated_table_size):
        self.id = id
        self.pe_mode = 0
        self.input_data = [0, -1, 0] # valid, idx, val
        self.allocated_table_size = allocated_table_size

    def get_mode(self):
        return self.pe_mode

    def receive(self, input_data = None, fetched_data = None):
        if input_data != None:
            self.input_data = input_data
        if fetched_data != None:
            self.fetched_data = fetched_data
    
    def tick(self):
        nnz_processed_h = 0

        # output request info
        req_valid = 0
        req_type = 0
        req_addr = 0
        req_data = (0, 0, 0) # idx, val, freq

        # input data
        valid = self.input_data[0]
        col_idx = self.input_data[1]
        val = self.input_data[2]

        # returned_data
        fetch_valid = self.fetched_data[0] #if the block is fetched from the hash table
        fetch_idx = self.fetched_data[1]
        fetch_val = self.fetched_data[2]
        fetch_freq = self.fetched_data[3]

        if self.pe_mode == 1:
            if fetch_valid != -1:
                if fetch_idx == -1:
                    fetch_idx = col_idx
                    fetch_val = val
                    req_valid = 1
                    req_type = 1
                    req_addr = self.hash_addr
                    req_data = [fetch_idx, fetch_val, fetch_freq]
                    self.pe_mode = 0
                    nnz_processed_h = 1
                elif fetch_idx == col_idx:
                    fetch_val += val
                    req_valid = 1
                    req_type = 1
                    req_addr = self.hash_addr
                    req_data = [fetch_idx, fetch_val, fetch_freq]
                    self.pe_mode = 0
                    nnz_processed_h = 1
                else:
                    self.hash_addr = (self.hash_addr + 1) & (self.ht_size_array - 1)
                    req_valid = 1
                    req_type = 0
                    req_addr = self.hash_addr
                self.fetched_data[0] = -1
        # process a new input
        elif self.pe_mode == 0: # pe is in idle mode
            if valid != 0:
                self.pe_mode = 1 # compute hash
                self.hash_addr = hash_index(col_idx, 0, self.allocated_table_size)
                req_valid = 1
                req_type = 0
                req_addr = self.hash_addr
                self.input_data[0] = 0
        else:
            print("bugged")
        
        self.clk += 1
        
        return [self.id, req_valid, req_type, req_addr, req_data], nnz_processed_h
