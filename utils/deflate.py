import os, sys
import scipy as sp
from ioBin import *


binFile = sys.argv[1]
extension = binFile[-3:]
if (extension !='bin'):
    sys.exit('\nExtension \'.'+extension+'\' not supported. Your file should be in binary format.\n')
asciiFile = binFile[:-3]+'txt'
print '\n deflating ', binFile, ' -> ', asciiFile,'...',; sys.stdout.flush()
D = readLargeBin(binFile)
sp.savetxt(asciiFile,D)
print ' done.\n'
