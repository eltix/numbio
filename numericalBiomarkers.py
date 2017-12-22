import scipy as sp
import sys, os
import numpy.linalg as lin
import matplotlib.pyplot as plt
import time
from utils_numBio import *
sp.set_printoptions(precision=3,threshold=2)

#-- Read input file and load data
plotResults = int(sys.argv[2]) # 0: False, 1: True
printStatusEvery = 1000 # Print convergence status every n iterations
caseName = sys.argv[1]
iterVal, l1Reg = readInputFile(caseName)
N, minIter, maxIter, tolStagnation, gradientStep = (iterVal[i] for i in range(5))
X, np = loadParamSamples(caseName, N)
Y, ny, nu, tau = loadModelOutputs(caseName, N)
penCons = 0.7; tolCons = 1.e-3
#
#-- This is expensive, therefore done once and for all
XTY = sp.dot(X.T,Y); YTYs = sp.dot(Y.T,Y); 
#
alphaCons = 1. + 1.e-4
B = [] # Biomarkers weights list
for index_param in range(np): # Loop on parameters
  YTY = penCons*YTYs.copy()
  u = sp.zeros((np+1,)); u[index_param] = 1.; u[-1] = 1.*penCons # u = [e_i, 1]
  # Initial guess
  x0 = sp.random.normal(0.,sp.sqrt(1./ny),(2*ny,))
  #print ' initial var = ', '{:.2e}'.format(sp.var(sp.dot(Y, x0[:ny]-x0[ny:])))
  y = x0.copy(); x = x0.copy(); tho=1.
  lam = l1Reg[index_param]/ny # l1-penalty parameter
  t = gradientStep/ny # Gradient step size (Nesterov)
  Jlist=[] # Cost Function list
  # Nesterov accelerated gradient iterations
  for i in range(maxIter):
    varConsSatisfied = True
    if (varCons(x, YTYs, N, ny)>tolCons):
      u[-1] *= alphaCons
      YTY *= alphaCons
      varConsSatisfied = False
    Jlist.append(J(x0, N, np, ny, u, lam, XTY, YTY))
    grad = DJ(y, N, np, ny, u, lam, XTY, YTY)
    dl = -t*grad
    x = y + dl
    # Constraints projection
    I = sp.nonzero(x<0)[0]
    if (len(I)>0): x[sp.nonzero(x<0)[0]] = 0.
    # Nesterov's trick and update
    th = -.5*tho**2 + .5*tho*sp.sqrt(4.+tho**2)
    s = (tho-tho**2)/(tho**2+th)
    y = x + s*(x-x0)
    tho = th
    x0 = x.copy()
    # Print convergence status
    if (i%printStatusEvery==0):
      print descentStatus(i, Jlist, grad, x, N, index_param, np, ny, u, XTY, YTY)
      #print 'penalization constraint =', u[-1]
    if (hasConverged(i, minIter, Jlist, tolStagnation) and varConsSatisfied):
      print 'Cost Function is stagnating';
      print ' Summary:'
      print '   distance to perfect covariance =', '{:.2e}'.format(lin.norm(sp.dot(XTY/N,x[:ny]-x[ny:])-u[:-1]))
      print '   L1 norm of w =','{:.2e}'.format(lin.norm(x[:ny]-x[ny:],1))
      print '   constraint error =', '{:.2e}'.format(sp.absolute(1.-sp.var(sp.dot(Y, x[:ny]-x[ny:]))))
      break
    if (i==maxIter-1):
      print 'Max iteration reached. Residual has not converged. Increase maxIter parameter.'
  B.append(x[:ny]-x[ny:])


#-- Plot results
if (plotResults):
  Z = sp.dot(Y,sp.array(B).T)
  A = (1./N)*sp.dot(X.T,Z)
  fig1 = plt.figure(figsize=[8,6])
  ax = fig1.add_subplot(111)
  ms = ax.matshow(A,vmin=0.,vmax=1.,cmap=plt.get_cmap('jet'))
  cbar = plt.colorbar(ms,ax=ax)
  sub = [(1,1),(1,1),(1,2),(2,2),(2,2),(2,3),(2,3),(4,2),(4,2)]
  fig2 = plt.figure(figsize=[8,8])
  for idx in range(np):
    ax = fig2.add_subplot(sub[np][0],sub[np][1],idx+1)
    ax.plot(X[:,idx],Z[:,idx],'k.',alpha=0.5)
    ax.set_ylim([-4.,4.])
  fig3 = plt.figure(figsize=[8,8])
  for idx in range(np):
    ax = fig3.add_subplot(sub[np][0],sub[np][1],idx+1)
    ax.plot(B[idx],'-o')
  plt.show()

#-- Save biomarkers weights
sp.savetxt('weights_'+caseName+'.out',sp.array(B+[nu,tau]))
print "\n All done."
