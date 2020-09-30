# Title         : EE5609:Matrix-theory Assignment

# Author        : Shreeprasad Bhat
#                 AI20MTECH14011
#                 M.Tech AI
#                 IIT Hyderabad

# Problem       : Write a python program to reduce any given matrix to row-reduced echelon form. Use this to compute inverse of matrix.

# Solution      :

import numpy as np
import copy
from sympy import Matrix # used for testing purpose


# ------------------------Utility functions-------------------------------------------
def iszero(elt) :
    if elt == 0 :
        return True
    else :
        return False

def is_square(matrix) :
# Input : 2D numpy array
# Output : True - if input is square matrix
#          False - if input is non-square matrix
    [nRows, nCols] = matrix.shape
    if nRows == nCols :
        return True
    return False

def is_singular(rref) :
# Input : row reduced echelon form of matrix of square matrix
# Output : True - if matrix is singular
#          False - if matrix is non-singular
    [nRows, nCOls] = rref.shape
    for iRow in range(nRows) :
        if np.all(rref[iRow, :] == 0) :
            return True
    return False


def iszerorow(row) :
    if np.all(row == 0) :
        return True
    else:
        return False

def do_row_exchange(matrix, row_num1,row_num2) :
# Input : matrix : 2D numpy array
#         row_num1, row_num2 : index of the rows to be interchanged
# Output : matrix : 2D numpy array with rows interchanged
    temp = copy.deepcopy(matrix[row_num1, :])
    matrix[row_num1, :] = copy.deepcopy(matrix[row_num2, :])
    matrix[row_num2, :] = copy.deepcopy(temp)
    return matrix

def do_row_reduction(rref, inverse, row_num, pivot_row_num, pivot_col_num) :
# Input : rref - 2D numpy array
#         inverse - 2D numpy array
#          row_num - index of row which is to be reduced
#          pivot_row_num, pivot_col_num - row and col of index of pivot element
    c = rref[row_num, pivot_col_num]
    rref[row_num, :] = rref[row_num, :] - c * rref[pivot_row_num, :]
    inverse[row_num, :] = inverse[row_num, :] - c * inverse[pivot_row_num, :]
    return [rref, inverse]

def make_pivot_elt_one(rref, inverse, pivot_row_num, pivot_col_num) :
    pivot = rref[pivot_row_num, pivot_col_num]
    rref[pivot_row_num, :] = rref[pivot_row_num, :] / pivot
    inverse[pivot_row_num, :] = inverse[pivot_row_num, :] / pivot
    return [rref, inverse]
# -------------------------------------------------------------------



# -----------------------------------------------------------------------
def find_rref(matrix) :
# Input : matrix - matrix whose row reduced echelon form is to be computed
# Output : rref - row reduced echelon form of input matrix
#          inverse - inverse of matrix, correct only for square matrix
    rref = copy.deepcopy(matrix)

    [nRows, nCols] = rref.shape
    inverse = np.identity(nRows)
    
    lastRow = nRows-1
    lastCol = nCols-1

    # to echelon form by row reduction
    curRow = 0
    for curCol in range(nCols) :
        if iszero(rref[curRow, curCol]) :
            nextRow = curRow + 1
            while nextRow <= lastRow :
                if not iszero(rref[nextRow, curCol]) : break
                nextRow += 1
            if nextRow <= lastRow : 
                rref = do_row_exchange(rref, curRow, nextRow)
                inverse = do_row_exchange(inverse, curRow, nextRow)
            else :
                continue
        [rref, inverse] = make_pivot_elt_one(rref, inverse, curRow, curCol)
        for nextRow in range(curRow + 1, lastRow + 1) :
            [rref, inverse] = do_row_reduction(rref, inverse, nextRow, curRow, curCol)
        if curRow == lastRow :
            if not iszero(rref[curRow, curCol]) :
                break
        else :
            curRow += 1
    
    # echelon form to row reduced echelon form by back substition
    prevPivotCol = lastCol
    for curPivotRow in range(lastRow,0,-1) :
        curPivotCol = 0
        while curPivotCol <= prevPivotCol and iszero(rref[curPivotRow, curPivotCol]) :
            curPivotCol += 1
        if curPivotCol <= lastCol :
            for prevRow in range(curPivotRow-1,-1,-1) :
                [rref, inverse] = do_row_reduction(rref, inverse, prevRow, curPivotRow, curPivotCol)
            prevPivotCol = curPivotCol

    return [rref, inverse]

# ---------------------------------------------------------------------------------------------




# ---------------------------Test functions----------------------------------------
def test_rref(matrix, computed_rref) :
    [actual_rref, c] = Matrix(matrix).rref()
    actual_rref = np.array(actual_rref.tolist()).astype('float64')
    if np.allclose(actual_rref, computed_rref) :
        print('\nrref - Pass!!!')
    else :
        print('\nrref - Fail')
        print('\nCorrect rref is\n')
        print(actual_rref)

def test_inverse(matrix, computed_inverse) :
    actual_inverse = np.linalg.inv(matrix)
    if np.allclose(actual_inverse, computed_inverse) :
        print('\nInverse - Pass!!!')
    else :
        print('\nInverse - Fail!!!')
        print('\nCorrect inverse is\n')
        print(actual_inverse)
# -------------------------------------------------------------------




# ---------------------------------------------------------------------------------------------
def display(matrix, rref, inverse) :
# display rref and inverse to terminal

    np.set_printoptions(precision=3, suppress=True)
    
    print('\nInput matrix is :\n')
    print(matrix)

    print('\nRow reduced echelon form of given matrix is :\n')
    print(rref)

    if not is_square(rref) :
        print('\nInverse does not exist for non-square matrix.')
    elif is_singular(rref) :
        print('\nInverse does not exist for singular matrix.')
    else :
        print('\nInverse of given matrix is :\n')
        print(inverse)
    
# ---------------------------------------------------------------------------------------------



# --------------------------------------testing------------------------------------------------
A1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
A2 = np.array([[0,0,1],[1,0,0],[0,1,0]])
A3 = np.array([[0,0,1,5],[0,1,2,1],[0,1,2,1]])
A4 = np.array([[1,3,1],[0,2,3],[34,-2,-1]])
A5 = np.array([[1,2,2],[-2,9,17],[1,0,0],[0,1,0]])
A6 = np.array([[1,1,1,1,1.0],[1,1,1,1,1],[1,1,1,1,1]])
A7 = np.array([[1,1,1,1,1.0],[1,1,2,2,2],[1,1,1,4,5]])
A8 = np.array([[1,1,1,1],[1,1,1,1],[1,2,3,4]])
A9 = np.array([[1,1,1],[1,1,2],[2,3,1]])

inputs = [A1, A2, A3, A4, A5, A6, A7, A8, A9] # add or remove matrix input here

count = 1
for A in inputs :
    matrix = A 

    matrix = matrix.astype('float64')

    [rref, inverse] = find_rref(matrix)

    print('\n===================================================')
    print('\nOutput - ',count)
    print('\n===================================================')

    display(matrix, rref, inverse)

    print('\n\nTest correctness using built-in methods :')
    
    test_rref(matrix, rref)
    if is_square(rref) and not is_singular(rref) : 
        test_inverse(matrix, inverse)
    print('\n===================================================')
    count += 1
# ---------------------------------------------------------------------------------------------