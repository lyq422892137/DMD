
from numpy import *
import gc


def cal_svd(A, svd_rank):
    m = A.shape[0]
    n = A.shape[1]

    # to calculate VT, the (n,n) left singular vector
    # S (m,n) diagonal matrix by sigma
    # U the right singular vector
    # a is the eigenvalue of A
    # v is the eigenvector of A

    U, sigma, VT = linalg.svd(A, full_matrices=False)
    V = VT.conj().T

    if svd_rank is 0:
        omega = lambda x: 0.56*x**3 - 0.95*x**2+1.82*x+1.43
        beta = divide(*sorted(A.shape))
        tau = median(sigma) * omega(beta)
        rank = sum(sigma>tau)
    elif svd_rank >0 and svd_rank<1:
        cumulative_energy = cumsum(sigma/sigma.sum())
        rank = searchsorted(cumulative_energy, svd_rank) +1
    elif svd_rank >=1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[1])
    else:
        rank = A.shape[1]

    U = U[:,:rank]
    V = V[:,:rank]
    sigma = sigma[:rank]

    return U, sigma, V

def svd_newMatrix(A,U,sigma,V,rank):
    # calculate new A
    S = mat(eye(rank) * sigma)
    A_new = dot(U, S).dot(V.T)
    # print(A_new.shape)
    # frobenius norm
    D = linalg.norm(A-A_new)
    print("svd error: "+ str(D))
    return A_new, D

def rsvd_newMatrix(A,U,sigma,V,rank,p=5):
    # calculate new A
    rank_new = rank + p
    S = mat(eye(rank_new) * sigma)
    A_new = dot(U, S).dot(V.T)

    # the error bound of rsvd:
    D = linalg.norm(A-A_new)
    print("rsvd error: " + str(D))
    return A_new, D


def rsvd(A,rank,p = 5,q = 1):
    # stage 1
    n = A.shape[1]
    rank_new = rank + p
    O = zeros((n,rank_new))
    for i in range(n):
        O[i,:] = random.normal(0,1,rank_new)

    Y = dot(A,O)
    Q, R = linalg.qr(Y)
    print("QR")
    print(Q.shape)
    print(R.shape)

    # stage 2
    # project A onto the low-dimensional subspace
    B = dot(Q.T,A)
    U_B, sigma, VT = linalg.svd(B, full_matrices=False)
    V = VT.conj().T
    U = dot(Q,U_B)
    del A,B,Y,n,rank_new
    gc.collect()
    return U, sigma, V





