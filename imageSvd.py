###################################################
#################
# try jpg
import numpy as np

from loadfile import loadimgs
A, X, Y = loadimgs(10)

###################################################

print(X.shape)
print(Y.shape)
print((X==Y).all())







