"""
 Title         : EE5609:Matrix-theory Programming Assignment 2

 Author        : Shreeprasad Bhat
                 AI20MTECH14011
                 M.Tech AI
                 IIT Hyderabad

 Problem       : 
 Write a program to find the Gram-Schmidt orthogonalization of the column vectors of a matrix.
 
 Given a matrix A, you have to compute a matrix B such that:
 B^TB = I, i.e., the columns are orthonormal, and
 
 Both A and B have the same column space. 
 You are only allowed to use basic numpy operations like transpose, norm, rank..

"""

import numpy as np

def has_zero_vector(M) :
    """
    Check for zero vector in input matrix
    """
    ncols = M.shape[1]
    for icol in range(ncols) :
        if np.all(M[:,icol] == 0) :
            return True
    return False

def compute_projection_matrix(v) :
    """
    Compute the projection matrix for vector v
    """
    v_t = np.transpose(v)
    dot_prod = np.dot(v.flatten(),v.flatten())
    P = 0
    if dot_prod != 0 :
        P = np.matmul(v,v_t) / dot_prod
    return P

def projection(u, v) :
    """
    Compute the projection of vector v on vector u
    """
    P = compute_projection_matrix(u)
    proj_v = np.dot(P,v)
    return proj_v

def compute_gram_schmidt(A):
    """
    This should compute the Gram Schmidt orthogonalization for the matrix A.
    The input will generally be an m-by-n real-valued matrix A
    The output should be a matrix B which satisfies the following properties:
       1. B^T B = I
       2. The ColumnSpace(B) = ColumnSpace(A)
    Both A and B should be np.array
    """
    nrows, ncols = A.shape
    
    B = np.zeros([nrows,ncols])

    # gram-schmidt orthogonalization
    for icol in range(ncols) :
        v = A[:,icol].reshape(len(A[:,icol]),1)
        for jcol in range(icol-1,-1,-1) :
            u = B[:,jcol].reshape(len(B[:,jcol]),1)
            v = v - projection(u, v)
        B[:, icol] = v.flatten()

    # gram-schmidt orthonormalization
    for icol in range(ncols) :
        u = B[:, icol]
        norm_u = np.linalg.norm(u)
        if norm_u != 0 :
            B[:, icol] = u/norm_u
    # we can remove all zero column, since they are linearly dependent on any column
    B = B[:,~ np.all(B == 0, axis=0)]
    return(B)

"""
Actual code begins here.
Make sure that the file testmatrices.npy is in the current directory

Each matrix is a numpy array
"""

test_matrices = np.load('testmatrices.npy',allow_pickle=True)
no_matrices = len(test_matrices)

for i in range(no_matrices):

    A = test_matrices[i].copy()
    if has_zero_vector(A) :
        print("Matrix has zero column, hence cannot be orthogonalized!!!")
        continue
    B = compute_gram_schmidt(A)
    print('Matrix ',i)
    testpassed = True
    if np.linalg.norm(np.matmul(B.transpose(),B) - np.eye(np.shape(B)[1])) > 0.0001:
        print('Test failed: B^TB is not identity')
        testpassed = False
    augmat = np.concatenate((A,B), axis=1)
    if not ( np.linalg.matrix_rank(augmat)==np.linalg.matrix_rank(A) and np.linalg.matrix_rank(A)==np.linalg.matrix_rank(B) ):
        print('Test failed: A and B do not have the same column space')
        testpassed = False
    if testpassed:
        print('Test passed')