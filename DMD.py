# inputs: X, Y, rank
# outputs:
# operators: phi, b, w, t，M(Koopman operator)

from svd import rsvd
from numpy import *

def rdmd(X, Y, D, rank=5, p=5, q=5):
    # compute X's svd:
    random.seed(7)
    rank_new = rank + p
    Ux, sigmax, Vx = rsvd(X, rank, p, q)
    Sx = mat(diag(sigmax)).I

    # compute M_hat
    M_hat = dot(Ux.T,Y).dot(Vx).dot(Sx)
    print("M")
    print(M_hat.shape)

    # a, b = np.linalg.eig(x)
    # a is eigenvalues, b is eigenvector
    # here， L = a, W = b
    L, W = linalg.eig(M_hat)

    # compute omega = ln(lambda)/delta t
    # for a standard video, delta t = 1
    # thus, omega = ln(L) = modes
    # omega = log(L)
    print("D:"+str(D))

    # compute phi (dynamic modes)  and B, the amplitudes
    phi = dot(Y,Vx).dot(Sx).dot(W)

    # b = []
    # for i in range(D.shape[1]):
    #     b.append(linalg.lstsq(phi,D[:,i])[0])
    # b = array(b)

    b = linalg.lstsq(phi, X[:, 0])[0]
    B = mat(eye(rank_new) * array(b))
    print("B")
    print(B.shape)

    V = ones((rank_new, D.shape[1]), dtype=complex)
    for i in range(len(L)):
        V[:, i] = L[i]
    for t in range(D.shape[1]):
        V[:, t] = V[:, t] ** t
    V2 = exp(V)
    print(V2.shape)
    print("phi:" + str(phi.shape))

    # print("phi:" + str((phi[0,0]==phi[0,0].all())))
    # print(phi)
    # print(phi.shape)


    return phi, B, V2

def compute_newD(phi,B,V):
    D_new = dot(phi, B).dot(V)
    return D_new









