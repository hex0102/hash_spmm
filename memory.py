from controller import *
import numpy as np
from utils import *
class Memory:
    
    n_channels = 0
    offset = 6 # 64Bye blocks
    c_bits = 0
    n_request = 0
    clk = 0
    matrix_space_bound = 0
    N_ROWS = 0
    
    def __init__(self, n_channels, matrix_space_bound, N_ROWS):
        self.n_channels = n_channels
        self.c_bits = int(np.log2(n_channels))
        #print("init with: "+str(self.c_bits)+" channel_bits")
        self.offset = 6
        self.controllers = []
        self.matrix_space_bound = matrix_space_bound
        self.N_ROWS = N_ROWS
        for i in range(self.n_channels):
            self.controllers.append(Controller(i))

    
    def enqueue(self, request):
        valid = request[0]
        if valid == 1: 
            self.n_request += 1
            addr = request[3]
            channel_id = (addr >> self.offset) & ((1<<self.c_bits) - 1)
            source_name = get_source(addr, self.matrix_space_bound, self.N_ROWS)
            #print("{}: enqueues {} to channel {} for {}".format(self.clk, addr, channel_id, source_name))

            #if(request[2]==129 and request[1]==0):
            #    print("enqueues {} to channel {}".format(request[3], channel_id))
            (self.controllers[channel_id]).enqueue(request)

    def tick(self):
        response = []
        for i in range(self.n_channels):
            response.append(self.controllers[i].tick())
        self.clk += 1
        return response
