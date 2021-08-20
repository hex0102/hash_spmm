from controller import *
import numpy as np

class Memory:
    
    n_channels = 0
    offset = 6 # 64Bye blocks
    c_bits = 0
    n_request = 0
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.c_bits = int(np.log2(n_channels))
        #print("init with: "+str(self.c_bits)+" channel_bits")
        self.offset = 6
        self.controllers = []
        for i in range(self.n_channels):
            self.controllers.append(Controller(i))

    
    def enqueue(self, request):
        valid = request[0]
        if valid == 1: 
            self.n_request += 1
            addr = request[3]
            channel_id = (addr >> self.offset) & ((1<<self.c_bits) - 1)
            #if(request[2]==129 and request[1]==0):
            #    print("enqueues {} to channel {}".format(request[3], channel_id))
            (self.controllers[channel_id]).enqueue(request)

    def tick(self):
        response = []
        for i in range(self.n_channels):
            response.append(self.controllers[i].tick())
        return response
