from utils import *
from hashcore import *


class HashSystem:

    clk = 0
    hashtable = None

    def __init__(self, n_pes, maximum_size, n_banks, unified_fifo):
        self.maximum_size = maximum_size
        self.unified_fifo = unified_fifo
        self.hashtable = np.zeros((maximum_size, 3)) # (id, value, freq)
        self.hashtable[:, 0] = -1
        self.n_banks = n_banks
        self.bank_busy = np.zeros((n_banks, 2))
        self.n_pes = n_pes

        self.request_q = []
        for i in range(self.n_pes):
            self.request_q.append([0, 0, 0, 0, 0])
        #self.pe_mode = np.zeros(self.n_pes) # 0 : idle, 1 : busy
        self.hashcore_array = []
        for i in range(self.n_pes):
            self.hashcore_array.append(HashCore(i, self.maximum_size))


    def reset_hashtable(self):
        self.hashtable[:, 0] = -1


    def send(self, bound):
        n_loaded = 0
        for i in range(self.n_pes):
            pe_status = self.hashcore_array[i].get_mode() 
            if( pe_status == 0 ):
                if len(self.unified_fifo)>0:
                    returned_result = self.unified_fifo.pop(0)
                    self.hashcore_array[i].receive([1, returned_result[0], returned_result[1]], None)
                    n_loaded += 1
                    if n_loaded > bound:
                        break
        return n_loaded


    def tick(self):
        n_processed = 0
        for i in range(self.n_pes):
            returned_request, processed_by_pe = self.hashcore_array[i].tick()
            if returned_request[1] == 1:
                self.request_q[i] = returned_request
            n_processed += processed_by_pe

        # load from the unified buffer

        access_bank_id = 0
        # hash table action
        for i in range(self.n_pes):
            curr_req = self.request_q[i]
            curr_pe_id = curr_req[0]
            curr_req_valid = curr_req[1]
            curr_req_type = curr_req[2]
            curr_req_addr = curr_req[3]
            curr_req_data = curr_req[4] 
            if curr_req_valid == 1: # valid request
                access_bank_id = curr_req_addr%self.n_banks
                if self.bank_busy[access_bank_id][0] == 0:
                    self.bank_busy[access_bank_id][0] = 1
                    if curr_req_type == 0: # read request
                        returned_entry = self.hashtable[curr_req_addr]
                        self.hashcore_array[curr_pe_id].receive(None, [1, returned_entry[0], returned_entry[1], returned_entry[2]])
                        self.bank_busy[access_bank_id][1] = 1
                        self.request_q[i][1] = 0
                    else: #write request  
                        self.hashtable[curr_req_addr][0] = curr_req_data[0]
                        self.hashtable[curr_req_addr][1] = curr_req_data[1]
                        self.hashtable[curr_req_addr][2] = curr_req_data[2]
                        self.bank_busy[access_bank_id][1] = 1
                        self.request_q[i][1] = 0
        
        
        # release the bank for next cycle access     
        for i in range(self.n_banks):
            if self.bank_busy[i][0] == 1 and self.bank_busy[i][1] == 1:
                self.bank_busy[i][0] = 0
                self.bank_busy[i][1] = 0
        
        return n_processed