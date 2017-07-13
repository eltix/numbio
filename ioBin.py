import os,sys
import scipy as sp
import struct

def writeLargeBin(fileName, A):

  binFile = open(fileName, mode='wb')
  b = struct.pack('ii',A.shape[0],A.shape[1])
  binFile.write(b) 
  A = sp.reshape(A,(A.shape[0]*A.shape[1],))
  for val in A:
    binFile.write(struct.pack('d', val))
  binFile.close()

def readLargeBin(fileName):

  f = open(fileName, mode='r')
  headFmt = 'ii'
  offset = struct.calcsize(headFmt)
  dataShape = struct.unpack(headFmt,f.read(offset))
  N = dataShape[0]; M = dataShape[1]
  D = sp.zeros((N,M))
  fmt = 'd'
  for i in range(N):
    D[i,:] = sp.array(struct.unpack(fmt*M,f.read(struct.calcsize(fmt)*M)))
  f.close()
  return D

