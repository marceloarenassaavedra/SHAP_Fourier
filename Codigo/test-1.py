from boolean_function import Boolean_function
from real_function import Real_function, inner_product, sum, scalar_product

f = Boolean_function("f1.txt")
g = Real_function(f)

p = Real_function(4, {})
p1 = Real_function(4, {1})
p2 = Real_function(4, {2})
p3 = Real_function(4, {3})
p4 = Real_function(4, {4})
p12 = Real_function(4, {1,2})
p13 = Real_function(4, {1,3})
p14 = Real_function(4, {1,4})
p23 = Real_function(4, {2,3})
p24 = Real_function(4, {2,4})
p34 = Real_function(4, {3,4})
p234 = Real_function(4, {2,3,4})
p134 = Real_function(4, {1,3,4})
p124 = Real_function(4, {1,2,4})
p123 = Real_function(4, {1,2,3})
p1234 = Real_function(4, {1,2,3,4})

h = sum(scalar_product(p, inner_product(g, p)),
    sum(scalar_product(p1, inner_product(g, p1)),
    sum(scalar_product(p2, inner_product(g, p2)),
    sum(scalar_product(p3, inner_product(g, p3)),
    sum(scalar_product(p4, inner_product(g, p4)),
    sum(scalar_product(p12, inner_product(g, p12)),
    sum(scalar_product(p13, inner_product(g, p13)),
    sum(scalar_product(p14, inner_product(g, p14)),
    sum(scalar_product(p23, inner_product(g, p23)),
    sum(scalar_product(p24, inner_product(g, p24)),
    sum(scalar_product(p34, inner_product(g, p34)),
    sum(scalar_product(p234, inner_product(g, p234)),
    sum(scalar_product(p134, inner_product(g, p134)),
    sum(scalar_product(p124, inner_product(g, p124)),
    sum(scalar_product(p123, inner_product(g, p123)),
    scalar_product(p1234, inner_product(g, p1234)))))))))))))))))

for i1 in {-1,1}:
    for i2 in {-1,1}:
        for i3 in {-1,1}:
            for i4 in {-1,1}:
                print(f'{i1} {i2} {i3} {i4} : {h.value([i1,i2,i3,i4])}')