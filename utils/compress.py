import os, sys
import scipy as sp
from ioBin import *


asciiFile = sys.argv[1]
extension = asciiFile[-3:]
if (extension !='txt'):
    sys.exit('\nExtension \'.'+extension+'\' not supported. Your file should be in ASCII format and the extension should be \'.txt\'.\n')
binFile = asciiFile[:-3]+'bin'
print '\n compressing ', asciiFile, ' -> ', binFile,'...',; sys.stdout.flush()
D = sp.loadtxt(asciiFile)
writeLargeBin(binFile,D)
print ' done.\n'
