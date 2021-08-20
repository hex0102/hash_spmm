# 64B/128 = 4 cycles = 4 nsB
# request is tuple(valid, type, id, addr)
# valid: 0/1
# type: 0 (read) | 1 (write)
# id: where it comes from
class Controller:
    
    clk = 0
    busy = 0
    latency = 40
    burst_cycle = 64*8/128
    id = 0

    # init method or constructor 
    def __init__(self, id):
        self.request_queue = []
        self.id = id
  
    def enqueue(self, request):
        #print("controller {} is equeuing {}".format(self.id, request[3]) )        
        self.request_queue.append((request, self.clk + self.latency))
    
    def tick(self):
        
        if(len(self.request_queue)!=0):
            top_request = self.request_queue[0]
            if(self.clk >= top_request[1] and self.busy == 0):
                self.busy = self.burst_cycle
                self.clk += 1
                return self.request_queue.pop(0)[0]
            else:
                if(self.busy>0):
                    self.busy -= 1
                self.clk += 1
                return (0, 0, 0, 0, 0)
        else:
            if(self.busy > 0):
                self.busy -= 1
            self.clk += 1
            return (0, 0, 0, 0, 0)
        

    
   
