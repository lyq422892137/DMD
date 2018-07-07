from loadfile import loadimgs
from svd import rsvd
A, X, Y = loadimgs(10)

###################################################

print(X.shape)
print(Y.shape)
print((X==Y).all())

Ux, sigmax, Ax = rsvd(X, 5, p = 5, q = 5)