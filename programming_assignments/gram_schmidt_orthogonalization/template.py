import numpy as np


def compute_gram_schmidt(A):
    """
    This should compute the Gram Schmidt orthogonalization for the matrix A.
    The input will generally be an m-by-n real-valued matrix A
    The output should be a matrix B which satisfies the following properties:
       1. B^T B = I
       2. The ColumnSpace(B) = ColumnSpace(A)
    Both A and B should be np.array
    """
    # Your code goes here
    
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
