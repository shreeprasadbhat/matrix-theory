from sympy import Matrix, Rational

Mat = Matrix([[1, 0], [0, 1], [Rational(5,2), 0]])

M = Mat * Mat.transpose()
print(M)
P, D = M.diagonalize(0)

print(P)
print(D)