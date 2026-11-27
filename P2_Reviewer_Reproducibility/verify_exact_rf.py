import sys
from itertools import product
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from rf_enclosure import range_bins, BIN, SAMPLE_SCALE, coefficient, ceil_power, enclose

def return_ranges(data):
    """Use three-point logic, with online initial half-cycle removal."""
    t=[data[0]]
    for i in range(1,len(data)-1):
        if data[i]!=t[-1] and (data[i]-t[-1])*(data[i+1]-data[i]) < 0:
            t.append(data[i])
    if t[-1]!=data[-1]:
        t.append(data[-1])
    stack=[]
    counts=[]
    for v in t:
        stack.append(v)
        while len(stack)>=3 and abs(stack[-3]-stack[-2])<=abs(stack[-2]-stack[-1]):
            r=abs(stack[-3]-stack[-2])
            if len(stack)==3:
                counts.append((1,r))
                del stack[0]
            else:
                counts.append((2,r))
                del stack[-3:-1]
    counts.extend((1,abs(b-a)) for a,b in zip(stack,stack[1:]))
    return counts

for tail in product(range(5),repeat=6):
    raw=[0,*tail]
    dedup=[raw[0]]
    for v in raw[1:]:
        if v!=dedup[-1]:
            dedup.append(v)
    ranges=return_ranges(dedup)
    base=np.array(raw,dtype=np.int64)*BIN
    lo,hi=range_bins(base,5)
    expect=np.zeros(5,dtype=np.int64)
    for w,r in ranges:
        expect[r]+=w
    np.testing.assert_array_equal(lo,expect)
    np.testing.assert_array_equal(hi,expect)
for order in [-1,0,1]:
    v=np.array([40*SAMPLE_SCALE,40*SAMPLE_SCALE+BIN,40*SAMPLE_SCALE+1,
                40*SAMPLE_SCALE+BIN+order,40*SAMPLE_SCALE],dtype=np.int64)
    lo,hi=range_bins(v,3)
    expect_lo=np.zeros(3,dtype=np.int64); expect_hi=expect_lo.copy()
    for w,r in return_ranges(v.tolist()):
        expect_lo[r//BIN]+=w
        expect_hi[(r+BIN-1)//BIN]+=w
    np.testing.assert_array_equal(lo,expect_lo)
    np.testing.assert_array_equal(hi,expect_hi)
c,d=coefficient()
assert c<d and d-c<F(1,10**20)
print('PASS: independent three-point comparison on 15,625 records and three lattice near-ties.')

