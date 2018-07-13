# inputs: X, Y, rank
# outputs:
# operators: phi, b, w, t，M(Koopman operator)

from svd import rsvd
from numpy import *
def rdmd(X, Y, rank=5,p=5,q=5):
    # compute X's svd:
    rank_new = rank+p
    Ux, sigmax, Vx = rsvd(X,rank,p,q)
    Sx = mat(eye(rank_new) * sigmax)
    M_hat = dot(Ux.T,Y).dot(Vx).dot(Sx.I)



