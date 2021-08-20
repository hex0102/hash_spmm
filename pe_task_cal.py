# this file is the main py for the cycle accurate simulation of hash-spmm
# baseline or proposed

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# import concurrent.futures
import multiprocessing
import numpy as np
import sys
import scipy.sparse
import scipy
from scipy import io
from scipy.sparse import *
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from scipy import sparse
from time import time
import math
from random import randint
import random
from memory import *
from utils import *
from pe import *
from hashpe import *

path_prefix = "/home/xinhe/hash_spmm/"

n_pes = 8




if __name__ == "__main__":
    outfilename = "ops_comparision.log"
    stats_file = open(outfilename, "a+")
    csr_ins, filename = load_sparse_matrix(path_prefix)
    m_height = csr_ins.shape[0]
    n_ops = np.zeros(m_height)
    n_ratios = np.zeros(int(math.ceil(m_height/n_pes)))
    for i in range(m_height):
        n_ops[i] = cal_nnz(csr_ins, i)
    for i in range(0, m_height, n_pes):
        cur_p = n_ops[i:i + n_pes]
        cur_p = cur_p[np.nonzero(cur_p)]
        ratio_n = 1
        if(cur_p.size>=1):
            ratio_n = np.max(cur_p)/np.min(cur_p)
        n_ratios[int(i/8)] = ratio_n

    print("{} {} {} {}".format(filename, csr_ins.nnz, csr_ins.nnz/csr_ins.shape[0], np.mean(n_ratios)), file=stats_file)


