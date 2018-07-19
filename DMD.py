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
    M_hat = compute_Mhat(Ux, Y, Vx, Sx)

    # compute L and W
    # L is eigenvalues, W is eigenvector
    L, W = compute_eig(M_hat)

    # compute phi (dynamic modes)  and B, the amplitudes
    phi = compute_phi(Y, Vx, Sx, W)

    B, b = compute_B(phi, rank_new, X[:,0])

    V = geneV(rank_new, D.shape[1], L)

    return phi, B, V

def compute_Mhat(U,Y,V,S):
    M_hat = dot(U.T, Y).dot(V).dot(S)
    return M_hat

def compute_eig(X):
    # a, b = np.linalg.eig(x)
    # a is eigenvalues, b is eigenvector
    # here， L = a, W = b
    L, W = linalg.eig(X)
    return L, W

def compute_phi(Y,V,S,W):
    phi = dot(Y, V).dot(S).dot(W)
    return phi

def compute_B(phi,rank_new, col):
    b = dot(phi.T,phi).I.dot(phi.T).dot(col)
    B = mat(eye(rank_new) * array(b))
    return B, b

# def geneV(rank_new, n, L):
#     V = ones((rank_new, n), dtype=complex)
#     for i in range(len(L)):
#         V[i, :] = L[i]
#         for t in range(n):
#             V[i, t] = V[i, t] ** t
#     return V

def geneV(rank_new, n, L):
    V = ones((rank_new, n), dtype=complex)
    fmode = log(L)
    for i in range(len(L)):
        V[i, :] = fmode[i]
        for t in range(n):
            V[i, t] = V[i, t] * t
    V = exp(V)
    return V

def geneV_fmode(rank_new, n, L):
    V1 = ones((rank_new, n), dtype=complex)
    V2 = ones((rank_new, n), dtype=complex)
    V3 = ones((rank_new, n), dtype=complex)
    fmode = log(L)
    # threshold = mean(fmode)
    threshold = median(fmode)
    # threshold = 0.001

    for i in range(len(L)):
        V1[i, :] = fmode[i]
        V2[i, :] = fmode[i]
        V3[i, :] = fmode[i]

        for t in range(n):
            V1[i, t] = V1[i, t] * t
            V2[i, t] = V2[i, t] * t
            V3[i, :] = V3[i, t] * t

    V1 = exp(V1)
    V2 = exp(V2)
    V3 = exp(V3)

    for j in range(len(L)):
        if abs(fmode[j]) < threshold:
            V2[j, :] = 0
        else:
            V1[j, :] = 0

    return V1, V2, V3


def compute_newD(phi,B,V):
    D_new = dot(phi, B).dot(V)
    return D_new

def object_extraction(X, Y, D, rank=5, p=5, q=5):
    random.seed(7)
    rank_new = rank + p
    Ux, sigmax, Vx = rsvd(X, rank, p, q)
    Sx = mat(diag(sigmax)).I
    M_hat = compute_Mhat(Ux, Y, Vx, Sx)
    L, W = compute_eig(M_hat)

    # compute phi (dynamic modes)  and B, the amplitudes
    phi = compute_phi(Y, Vx, Sx, W)

    B, b = compute_B(phi, rank_new, X[:, 0])

    V1, V2, V3 = geneV_fmode(rank_new, D.shape[1], L)

    return phi, B, V1, V2, V3





