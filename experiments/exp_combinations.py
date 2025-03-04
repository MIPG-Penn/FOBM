#%%
import math

n = 7669
# n = 10
p = 0

pi = 0
for r in range(1, n + 1):
    pr = math.factorial(n)//(math.factorial(r)*math.factorial(n-r))
    p += pr
    
    print(f"{r} {pr}")

print(f"{p}")

pt_ns = p
pt_s = pt_ns // 1000000000
# print(pt_s)

pt_h = pt_s // 60
# print(pt_h)

pt_d = pt_h // 24
# print(pt_d)

pt_y = pt_d // 365
# print(pt_y)

pt_b = pt_y // 1000000000
# print(pt_m)

pow = len(str(pt_b))-1

print(f"With {n} features and assuming 1 nanosecond per combination it would take 10^{pow} billion years")