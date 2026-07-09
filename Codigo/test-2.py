from boolean_function import Boolean_function
from real_function import Real_function, print_Fourier_coefficients

f = Boolean_function("f1.txt")
g = Real_function(f)

x = [-1, -1, -1, -1]

print_Fourier_coefficients(g)