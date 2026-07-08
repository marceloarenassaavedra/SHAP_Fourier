from boolean_function import Boolean_function
from real_function import Real_function, print_SHAP_spectrum, print_SR_spectrum

f = Boolean_function("f2.txt")
g = Real_function(f)

x = [-1, -1, -1, -1]
S = {1, 2}

print(f'SHAP score for feature 1: {g.SHAP(x,1)}\n')
print_SHAP_spectrum(g, x, 1)
print('')
print_SR_spectrum(g, x, S)
