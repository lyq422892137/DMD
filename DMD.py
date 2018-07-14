# inputs: X, Y, rank
# outputs:
# operators: phi, b, w, t，M(Koopman operator)

from svd import rsvd
from numpy import *

def rdmd(X, Y, D, rank=5,p=5,q=5):
    # compute X's svd:
    random.seed(7)
    rank_new = rank+p
    Ux, sigmax, Vx = rsvd(X,rank,p,q)
    Sx = mat(eye(rank_new) * sigmax)

    # compute M_hat
    M_hat = dot(Ux.T,Y).dot(Vx).dot(Sx.I)

    # compute upper lambda L
    # a, b = np.linalg.eig(x)
    # a is eigenvalues, b is eigenvector
    # here， L = a, W = b
    L, W = linalg.eig(M_hat)

    # compute omega = ln(lambda)/delta t
    # for a standard video, delta t = 1
    # thus, omega = ln(L)
    omega = log(L)
    # print("W:"+ str(W)) W没问题
    # print("Vx:" + str(Vx)) Vx no problem
    # print("Sx.I:" + str(Sx.I)) no problem but the values are very big
    # compute phi and B, the amplitudes
    phi = dot(Y,Vx).dot(Sx.I).dot(W)
    print("phi:" + str((phi[0,0]==phi[0,0].all())))
    print(phi)
    b = dot(phi.T,phi).I.dot(phi.T).dot(X[:,0])
    B = mat(eye(rank_new) * array(b))

    return phi, B, omega

def compute_V(n,k,omega):
    V = ones((n, k), dtype=complex)
    for i in range(len(omega)):
        V[:, i] = omega[i]
    return V

# def computeImags(omega, phi, B, n, threshold = 1):
#     k = len(omega)
#     m = B.shape[0]
#     V = compute_V(n, k, omega)
#     l = []
#     s = []
#     l_count = []
#     s_count = []
#
#     for i in range(len(omega)):
#         if sqrt(power(omega[i],2)) < threshold:
#             l.append(omega[i])
#             l_count.append(i)
#         else:
#             s.append(omega[i])
#             s_count.append(i)
#

    # l = []
    # l_count = []
    # s = []
    # s_count = []
    # V = compute_V(n, k, omega, -1)
    #
    # for i in range(len(omega)):
    #     if power(omega[i],2) < threshold:
    #         l.append(B[i]*phi[i]*exp(V[]))
    #         l_count.append(i)
    #     else:
    #         s.append(omega[i])
    #         s_count.append(i)
    #
    print(len(omega))
    print(len(s))
    print(s_count)
    #
    # return l, s, l_count, s_count

def compute_newD(phi,B,n,k,omega):
    # V = compute_V(n,k,omega)
    V = ones((k, n), dtype=complex)
    for i in range(len(omega)):
        V[:, i] = omega[i]
    for t in range(n):
        V[:,i] = V[:,i] * t
    V = exp(V)
    print("V:" +str((V[:, 0] == V[:, 3]).all()))
    print(V.shape)
    print(omega[4])
    print(V[0,4])
    print("phi:" +str(phi.shape))
    # D_new = dot(phi, B).dot(V)
    D_new = dot(phi, B).dot(V)
    print("-----------------------")
    print(D_new)
    print("-----------------------")
    return D_new

#
# def compute_background(D_new,s_count, n, k ,omega,phi, B):
#     V = compute_V(n, k, omega, s_count)
#     L = dot(phi, B).dot(V.T)
#     return L
#
# def compute_foreground(D_new,l_count):
#     S = delete(D_new, l_count, 1)
#     return S










