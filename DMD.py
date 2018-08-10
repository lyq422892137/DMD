# author: yl90

# for rsvd computation
from svd import rsvd
# for mathematical calculation
from numpy import *
# for memory cleaning
import gc


# rdmd: randomized dynamic mode decomposition
# inputs:
# D: the whole input matrix
# X: the first frame matrix from 0 to n-2
# Y: the second frame matrix from 1 to n-1
# p: the oversample parameter
# q: the power used for calculate rsvd
# outputs:
# phi: the dynamic mode
# B: the diagonal matrix of the amplitudes
# V: the vandermonde matrix
def rdmd(X, Y, D, rank=5, p=5, q=5):
    # compute X's svd:
    random.seed(7)
    rank_new = rank + p
    Ux, sigmax, Vx = rsvd(X, rank, p, q)
    Sx = mat(diag(sigmax,reverse=True)).I

    # compute M_hat
    M_hat = compute_Mhat(Ux, Y, Vx, Sx)

    # compute L and W
    # L is eigenvalues, W is eigenvector
    L, W = compute_eig(M_hat)

    # compute phi (dynamic modes)
    phi = compute_phi(Y, Vx, Sx, W)

    # compute B
    B, b = compute_B(phi, rank_new, X[:,0])

    # compute the vandermonde matrix
    V = geneV(rank_new, D.shape[1], L)

    # memory cleaning
    del M_hat, L, W, X, Y, D, Ux, Sx, Vx, b
    gc.collect()

    return phi, B, V

def compute_Mhat(U,Y,V,S):
    M_hat = dot(U.T, Y).dot(V).dot(S)
    del U, Y, V, S
    gc.collect()
    return M_hat

# a, b = np.linalg.eig(x)
# a is eigenvalues, b is eigenvector
# here， L = a, W = b
def compute_eig(X):
    L, W = linalg.eig(X)
    del X
    gc.collect()
    return L, W

def compute_phi(Y,V,S,W):
    phi = dot(Y, V).dot(S).dot(W)
    del Y, V, S, W
    gc.collect()
    return phi

def compute_B(phi,rank_new, col):
    b = dot(phi.T,phi).I.dot(phi.T).dot(col)
    B = mat(eye(rank_new) * array(b))
    del phi, rank_new, col
    gc.collect()
    return B, b

def geneV(rank_new, n, L):
    V = zeros((rank_new, n), dtype=complex)
    fmode = log(L)
    for i in range(len(L)):
        V[i, :] = fmode[i]
        for t in range(n):
            V[i, t] = V[i, t] * t
    V = exp(V)
    del rank_new, n, L, fmode
    gc.collect()
    return V

# use Fourier Mode to generate the vandermonde matrix
# separate the matrix into three kinds:
# (1) V2 for objects whose fourier modes are bigger than threshold
# (2) V1 for background whose fourier modes are smaller than/equal to threshold
# (3) V3 the new images with the entire fourier modes
def geneV_fmode(rank_new, n, L,m):
    V1 = zeros((rank_new, n), dtype=complex)
    V2 = zeros((rank_new, n), dtype=complex)
    V3 = zeros((rank_new, n), dtype=complex)
    fmode = log(L)
    # threshold = mean(abs(fmode))/10
    # print(fmode)
    threshold = 1/sqrt(max(n,m))

    for i in range(len(L)):
        V1[i, :] = fmode[i]
        V2[i, :] = fmode[i]
        V3[i, :] = fmode[i]

        for t in range(n):
            V1[i, t] = V1[i, t] * t
            V2[i, t] = V2[i, t] * t
            V3[i, t] = V3[i, t] * t

    V1 = exp(V1)
    V2 = exp(V2)
    V3 = exp(V3)

    for j in range(len(L)):
        if abs(fmode[j]) <= threshold:
            V2[j,:] = 0
        else:
            V1[j,:] = 0

    # V2 = V2*10

    del fmode, L, rank_new, threshold
    gc.collect()

    return V1, V2, V3

# compute the new image matrix by the rdmd formula
def compute_newD(phi,B,V):
    D_new = dot(phi, B).dot(V)
    del phi, B, V
    gc.collect()
    return D_new

# the method arrange the rdmd computation and background/foreground extraction process:
# inputs:
# D: the whole input matrix
# X: the first frame matrix from 0 to n-2
# Y: the second frame matrix from 1 to n-1
# p: the oversample parameter
# q: the power used for calculate rsvd
# rank: the target rank for decompostion

# outputs:
# parameters for a new image matrix computation:
# phi: the dynamic mode
# B: the diagonal matrix of the amplitudes
# V1, V2, V3: background, foreground, and the new image matrices individually
def object_extraction(X, Y, D, rank, p, q):
    random.seed(7)
    rank_new = rank + p
    Ux, sigmax, Vx = rsvd(X, rank, p, q)
    Sx = mat(diag(sigmax)).I
    M_hat = compute_Mhat(Ux, Y, Vx, Sx)
    L, W = compute_eig(M_hat)

    # compute phi (dynamic modes)  and B, the amplitudes
    phi = compute_phi(Y, Vx, Sx, W)

    B, b = compute_B(phi, rank_new, X[:, 0])

    V1, V2, V3 = geneV_fmode(rank_new, D.shape[1], L, X.shape[0])

    del X, Y, D, rank, p, q
    gc.collect()

    return phi, B, V1, V2, V3

# compute errors between groundtruth and objects
def errorComputation(Objects, G, x_pix, y_pix):
    ImgNo = Objects.shape[1]
    Error = G - Objects
    error = np.sum(np.sum(Error))/ImgNo/x_pix/y_pix
    print("sub-error: " + str(error))
    return error, Error




