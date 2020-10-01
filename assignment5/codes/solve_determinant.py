from sympy import Matrix, symbols, Rational, solve

h = symbols('h')

M = Matrix([[6, h, Rational(22,2)], [h, 12, Rational(31,2)],[Rational(22,2), Rational(31,2), 20]])
eq = M.det()

sol = solve(eq, dict=True)

print(sol)
