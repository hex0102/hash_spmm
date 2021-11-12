from utils import *
from hashcore import *


class HashSystem:

    clk = 0
    hashtable = None

    def __init__(self, n_pes, maximum_size, n_banks, unified_fifo, start_temp, external_hashtable, external_request):
        self.maximum_size = maximum_size
        self.unified_fifo = unified_fifo
        self.hashtable = np.zeros((maximum_size, 4)) # (id, value, freq, dirty)
        #self.hashtable_dirty = np.zeros(maximum_size)
        self.hashtable[:, 0] = -1
        self.n_banks = n_banks
        self.bank_busy = np.zeros((n_banks, 2))
        self.n_pes = n_pes
        self.START_TEMP = start_temp
        self.external_hashtable = external_hashtable
        self.debug_counter = [np.zeros(1), [], [], []] #np.zeros(1)
        self.clk = 0
        self.on_chip_table_size = 0
        self.off_chip_table_size = 0
        self.external_request = external_request
        self.request_q = []
        for i in range(self.n_pes):
            self.request_q.append([0, 0, 0, 0, 0, 0, 0, 0])
        #self.pe_mode = np.zeros(self.n_pes) # 0 : idle, 1 : busy
        self.hashcore_array = []
        for i in range(self.n_pes):
            self.hashcore_array.append(HashCore(i, self.maximum_size, self.hashtable, self.debug_counter, self.external_request))

        self.bank_granted = []
        for i in range(self.n_banks):
            self.bank_granted.append([])

    def set_external_request(self, external_request):
        self.external_request = external_request
        for i in range(self.n_pes):
            self.hashcore_array[i].set_external_request(external_request)
            
    def set_table_size(self, on_chip_size, off_chip_size):
        self.on_chip_table_size = on_chip_size
        self.off_chip_table_size = off_chip_size
        for i in range(self.n_pes):
            self.hashcore_array[i].set_table_size(on_chip_size, off_chip_size)


    def reset_hashtable(self):
        self.hashtable[:2*self.on_chip_table_size , 0] = -1
        self.hashtable[:2*self.on_chip_table_size , 1] = 0
        self.hashtable[:2*self.on_chip_table_size , 2] = 0
        self.hashtable[:2*self.on_chip_table_size , 3] = 0
        self.external_hashtable[:2*self.off_chip_table_size, 0] = -1
        self.external_hashtable[:2*self.off_chip_table_size, 1] = 0
        self.external_hashtable[:2*self.off_chip_table_size, 2] = 0
        self.external_hashtable[:2*self.off_chip_table_size, 3] = 0        

    def receive(self, received_data_tuple, pe_id):
        self.hashcore_array[pe_id].receive(None, [1, received_data_tuple[0], received_data_tuple[1], received_data_tuple[2], received_data_tuple[3]])

    def send(self, bound):#send data from unified fifo to PEs
        n_loaded = 0
        for i in range(self.n_pes):
            pe_status = self.hashcore_array[i].get_mode() 
            write_complete = self.hashcore_array[i].get_write_status() 
            if( pe_status == 0 and write_complete == 1):
                if len(self.unified_fifo)>0:
                    returned_result = self.unified_fifo.pop(0)
                    self.hashcore_array[i].receive([1, returned_result[0], returned_result[1]], None)
                    n_loaded += 1
                    if n_loaded > bound:
                        break
        return n_loaded


    def tick(self):
        n_processed = 0
        #missing = np.setdiff1d(np.unique(np.array(self.debug_counter[2])[:,0]), self.hashtable[(self.hashtable[:,0]!=-1).nonzero()[0], 0])
        for i in range(self.n_pes):
            returned_request, processed_by_pe = self.hashcore_array[i].tick()
            if returned_request[0] == 0 and returned_request[2] == 1:
                self.request_q[i][0:-1] = returned_request
                self.request_q[i][-1] = 0
            elif returned_request[0] == 1 and returned_request[1] == 1:
                self.request_q[i][0:-1] = returned_request
                self.request_q[i][-1] = 0
            #n_processed += processed_by_pe

        list_external_reqs = []

        access_bank_id = 0

        for i in range(self.n_pes):
            curr_req = self.request_q[i]
            if curr_req[0] == 0 and curr_req[-1] != 1: # access shared on-chip hash table
                curr_req_valid = curr_req[2]
                curr_req_addr = curr_req[4]
                if curr_req_valid == 1:
                    access_bank_id = curr_req_addr%self.n_banks
                    self.bank_granted[access_bank_id].append(i)
                    self.request_q[i][-1] = 1
        
        for i in range(self.n_banks):                
            if len(self.bank_granted[i]):
                curr_pe_id = self.bank_granted[i].pop(0)
                curr_req = self.request_q[curr_pe_id]
                curr_pe_id = curr_req[1]
                curr_req_valid = curr_req[2]
                curr_req_type = curr_req[3]
                curr_req_addr = curr_req[4]
                curr_req_data = curr_req[5]
                if curr_req_type == 0: # read request
                    returned_entry = self.hashtable[curr_req_addr]
                    self.hashcore_array[curr_pe_id].receive(None, [1, returned_entry[0], returned_entry[1], returned_entry[2], returned_entry[3]])
                    self.hashtable[curr_req_addr][3] = 1
                    #self.bank_busy[access_bank_id][1] = 1
                    self.request_q[curr_pe_id][2] = 0
                else: #write request
                    n_processed += 1  
                    self.hashcore_array[curr_pe_id].write_done(1)
                    self.hashtable[curr_req_addr][0] = curr_req_data[0]
                    self.hashtable[curr_req_addr][1] = curr_req_data[1]
                    self.hashtable[curr_req_addr][2] = curr_req_data[2]
                    #self.bank_busy[access_bank_id][1] = 1
                    self.request_q[curr_pe_id][2] = 0

        for i in range(self.n_pes):
            curr_req = self.request_q[i]
            if curr_req[0] == 1: # access external memory #[self.fetch_dest, req_valid, req_type, PE_START_ID + self.id,  int(self.START_TEMP) +  req_addr*HASH_ITEM_SIZE, 1, data]       
                curr_req_valid = curr_req[1]
                curr_req_type = curr_req[2]
                curr_pe_id = curr_req[3]
                curr_req_addr = curr_req[4]
                curr_req_n = curr_req[5]
                curr_req_data = curr_req[6]
                if curr_req_valid == 1:
                    #curr_request = (1, 0, self.id,  int(self.START_A_INDICE_ADDR + a_row_start_idx*8) \+ self.cur_a_row_nnz*8, local_n)
                    curr_req_addr += int(self.START_TEMP)
                    external_request = (1, curr_req_type, curr_pe_id, curr_req_addr, curr_req_n, curr_req_data)
                    list_external_reqs.append(external_request)
                    self.request_q[i][1] = 0
                    if curr_req_type == 1:
                        n_processed += 1
                        self.hashcore_array[curr_pe_id - PE_START_ID].write_done(1)               
        '''
        # hash table action
        for i in range(self.n_pes):
            curr_req = self.request_q[i]
            if curr_req[0] == 0: # access shared on-chip hash table
                curr_pe_id = curr_req[1]
                curr_req_valid = curr_req[2]
                curr_req_type = curr_req[3]
                curr_req_addr = curr_req[4]
                curr_req_data = curr_req[5]
                #curr_req_dest = curr_req[5] 
                if curr_req_valid == 1: # valid request
                    access_bank_id = curr_req_addr%self.n_banks
                    if self.bank_busy[access_bank_id][0] == 0:
                        self.bank_busy[access_bank_id][0] = 1
                        if curr_req_type == 0: # read request
                            returned_entry = self.hashtable[curr_req_addr]
                            self.hashcore_array[curr_pe_id].receive(None, [1, returned_entry[0], returned_entry[1], returned_entry[2], returned_entry[3]])
                            self.hashtable[curr_req_addr][3] = 1
                            self.bank_busy[access_bank_id][1] = 1
                            self.request_q[i][2] = 0
                        else: #write request
                            n_processed += 1  
                            self.hashcore_array[curr_pe_id].write_done(1)
                            self.hashtable[curr_req_addr][0] = curr_req_data[0]
                            self.hashtable[curr_req_addr][1] = curr_req_data[1]
                            self.hashtable[curr_req_addr][2] = curr_req_data[2]
                            self.bank_busy[access_bank_id][1] = 1
                            self.request_q[i][2] = 0
            elif curr_req[0] == 1: # access external memory #[self.fetch_dest, req_valid, req_type, PE_START_ID + self.id,  int(self.START_TEMP) +  req_addr*HASH_ITEM_SIZE, 1, data]       
                curr_req_valid = curr_req[1]
                curr_req_type = curr_req[2]
                curr_pe_id = curr_req[3]
                curr_req_addr = curr_req[4]
                curr_req_n = curr_req[5]
                curr_req_data = curr_req[6]
                if curr_req_valid == 1:
                    #curr_request = (1, 0, self.id,  int(self.START_A_INDICE_ADDR + a_row_start_idx*8) \+ self.cur_a_row_nnz*8, local_n)
                    curr_req_addr += int(self.START_TEMP)
                    external_request = (1, curr_req_type, curr_pe_id, curr_req_addr, curr_req_n, curr_req_data)
                    list_external_reqs.append(external_request)
                    self.request_q[i][1] = 0
                    if curr_req_type == 1:
                        n_processed += 1
                        self.hashcore_array[curr_pe_id - PE_START_ID].write_done(1)         
        # release the bank for next cycle access     
        for i in range(self.n_banks):
            if self.bank_busy[i][0] == 1 and self.bank_busy[i][1] == 1:
                self.bank_busy[i][0] = 0
                self.bank_busy[i][1] = 0
        '''
        self.clk += 1
        
        return n_processed, list_external_reqs