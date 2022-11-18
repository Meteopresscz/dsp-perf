#!/usr/bin/env python3

import numpy as np
from icecream import ic
import sys
import time

data = np.fromfile("input", dtype=np.uint32)
iterations = 10000

def data2pol(d):
  pol = d[::2]

  # array is now not contiguous, so we cannot view
  pol = pol.copy()

  data = pol.view(np.uint16)

  pol = (data & 0xEFFF) | ((data & 0xE000)>>1)

  cfile = pol.view(np.int16).astype(np.float32).view(np.complex64)
  return cfile

x = time.time()
for i in range(iterations):
  pol1 = data2pol(data)
  pol2 = data2pol(data[1:])
delta = time.time()-x

print("%i iterations in %f s = %f us/iteration"%(iterations, delta, delta*1000000/iterations))

pol1.tofile("pol1.py.bin")
pol2.tofile("pol2.py.bin")
