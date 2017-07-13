import scipy as sp
import sys, os
import numpy.linalg as lin
from ioBin import *

def Pmat(x, N, XTY, YTY):
  M = (1./N)*sp.vstack(( XTY,sp.dot(x.T,YTY) ))
  return M

def Qmat(x, N, np, ny, YTY):
  M = (1./N)*sp.vstack(( sp.zeros((np,ny)) ,sp.dot(x.T,YTY) ))
  return M

def trueCost(x, N, np, ny, u, XTY, YTY):
  xp   = x[:ny]; xm = x[ny:]; beta = xp-xm
  K    = Pmat(xp, N, XTY, YTY) - Qmat(xm, N, np, ny, YTY)
  cost = lin.norm(sp.dot(K,beta)-u)
  return cost

def J(x, N, np, ny, u, lam, XTY, YTY):
  xp   = x[:ny]; xm = x[ny:]; beta = xp-xm
  K    = Pmat(xp, N, XTY, YTY) - Qmat(xm, N, np, ny, YTY)
  cost = .5*lin.norm(sp.dot(K,beta)-u)**2 + lam*(sp.sum(xp)+sp.sum(xm))
  return cost

def DJ(x, N, np, ny, u, lam, XTY, YTY):
  xp = x[:ny]; xm = x[ny:]; beta = xp-xm
  P  = Pmat(xp, N, XTY, YTY); Q = Qmat(xm, N, np, ny, YTY)
  P2 = P.copy(); P2[-1,:] *=2.
  K  = P-Q
  r  = sp.dot(K,beta)-u
  gradp = sp.dot(( P2-2.*Q ).T,r)
  gradm = sp.dot(( 2.*Q-P-Qmat(xp, N, np, ny, YTY) ).T,r)
  grad  = sp.hstack((gradp,gradm)) + lam*sp.ones(x.shape)
  return grad

def readInputFile(caseName):
  inputFile = open('./data/'+caseName+'/numbio.in','r')
  iterVal = [float(v) for v in inputFile.readline().split()]
  l1Reg   =  [float(v) for v in inputFile.readline().split()]
  for i in range(3): iterVal[i] = int(iterVal[i])
  inputFile.close()
  return iterVal, l1Reg

def loadParamSamples(caseName, N):
  X  = sp.loadtxt('./data/'+caseName+'/params.txt')
  if (N>X.shape[0]):
    sys.exit("Error: Nsamples asked larger than available. Please modify your numbio.in file.")
  else: X = X[:N,:]
  X  = (X-sp.mean(X,0))/sp.std(X,0)
  np = X.shape[1]
  return X, np

def loadModelOutputs(caseName, N):
  try:
    G   = readLargeBin('./data/'+caseName+'/outputs.bin')
  except IOError as e:
    print "I/O error({0}): {1}".format(e.errno, e.strerror)
    print "Trying a txt format instead:"
    try:
        G   = sp.loadtxt('./data/'+caseName+'/outputs.txt')
    except IOError as e:
      print "I/O error({0}): {1}".format(e.errno, e.strerror)

  if (N>G.shape[0]):
    sys.exit("Error: Nsamples asked larger than available. Please modify your numbio.in file.")
  else: G = G[:N,:]
  ny  = G.shape[1]
  nu  = sp.mean(G,0)
  tau = sp.std(G,0)#.max() * sp.ones((ny,))
  Y   = (G - nu)/tau
  Z = sp.sum(sp.isnan(Y))
  if Z>0: print 'WARNING: Y has NaN entries in loadModelOutputs (utils_numBio.py)'

  return Y, ny, nu, tau

def descentStatus(i, Jlist, grad, x, N, np, ny, u, XTY, YTY):
  toPrint =  'Iter # '+str(i)+': J(x)='+'{:.3e}'.format(Jlist[-1])+' | |DJ(x)|='
  toPrint += '{:.1e}'.format((lin.norm(grad)))+' | ||x||_1='
  toPrint += '{:.2e}'.format(lin.norm(x[:ny]-x[ny:],1))
  toPrint += ' | cost='+'{:.2e}'.format(trueCost(x, N, np, ny, u, XTY, YTY))
  return toPrint

def hasConverged(i, minIter, Jlist, tolStagnation):
  return (i>minIter) and ((sp.absolute(Jlist[-1]-Jlist[-2])/(Jlist[-2]))<tolStagnation)
