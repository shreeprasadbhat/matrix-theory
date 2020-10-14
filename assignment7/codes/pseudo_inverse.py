from sympy import Matrix, Rational

Mat = Matrix([[1, 0], [0, 1], [Rational(5,2), 0]])

p_inverse = Mat.pinv()

print(p_inverse)