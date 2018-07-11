
from numpy import *


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

    # print(sigma.shape)
    # # print(S)
    # print("U " + str(U.shape))
    # print("V " + str(V.shape))
    return U, sigma, V

def svd_newMatrix(A,U,sigma,V,rank):
    # calculate new A
    S = mat(eye(rank) * sigma)
    A_new = dot(U, S).dot(V.T)
    # print(A_new.shape)
    # frobenius norm
    D = linalg.norm(A-A_new)
    print("svd: "+ str(D))
    return A_new, D

def rsvd_newMatrix(A,U,sigma,V,rank,p=5):
    # calculate new A
    rank_new = rank + p
    S = mat(eye(rank_new) * sigma)
    A_new = dot(U, S).dot(V.T)
    # print(A_new.shape)
    # the error bound of rsvd:
    D = linalg.norm(A-A_new)
    print("rsvd: " + str(D))
    return A_new, D


def rsvd(A,rank,p = 5,q = 5):
    # stage 1
    n = A.shape[1]
    rank_new = rank + p
    O = zeros((n,rank_new))
    for i in range(n):
        O[i,:] = random.normal(0,1,rank_new)

    # print(O)

    # Y = power(dot(A,A.T),q).dot(A).dot(O) // memory error
    Y = dot(A,O)
    # print(Y.shape)
    Q, R = linalg.qr(Y)
    # A_new = dot(Q,Q.T).dot(A)
    # D = linalg.norm(A - A_new)
    # print(D)

    # stage 2
    # project A onto the low-dimensional subspace
    B = dot(Q.T,A)
    # print(B.shape)
    U_B, sigma, VT = linalg.svd(B, full_matrices=False)
    # print(U_B.shape)
    V = VT.conj().T
    U = dot(Q,U_B)
    return U, sigma, V

# memory error when imgNp = 2
# def computeM_1(X,Y,rank):
#     # U, sigma, V = rsvd(X,rank=rank)
#     print("-----------------------------")
#     # U, sigma, V = cal_svd(A,rank)
#     U, sigma, VT = linalg.svd(A, full_matrices=False)
#     V = VT.conj().T
#     print("V: " + str(V.shape))
#     print("U: " + str(U.shape))
#     print("sigma: " + str(len(sigma)))
#     S = mat(eye(rank) * sigma)
#     print(S.shape)
#     S = S.I
#     print(S.shape)
#     M = dot(Y,V.T)
#     print(M.shape)
#     # .dot(S).dot(U.T)
#     print(M.shape)
    # U2, sigma2, V2 = cal_svd(1/X,0)
    # print(U1.shape)
    # print(U2.shape)
    # print(sigma1)
    # print(sigma2)
    # print(V1.shape)
    # print(V2.shape)
    # print(X.shape)
    # print((1/X).shape)

def rDMD(D,X,Y,k):
    U, sigma, V = rsvd(X,rank=k)
    print(len(sigma))
    S = diag(sigma)
    print(S.shape)
    M = dot(U.T,Y).dot(V).dot(S)
    W, l = linalg.eig(M)
    F = mat(dot(Y,V).dot(S).dot(W))
    print(F.shape)
    b = linalg.lstsq(F,mat(X[:,0]))
    V = vander(l)






