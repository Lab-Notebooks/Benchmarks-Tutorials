import numpy

rhoLiq = 1000
rhoGas = 100
muLiq = 10
muGas = 1
g = .98
sig = 24.5

r = 0.25
l = 2*r
u = numpy.sqrt(g*l)

Re = rhoLiq*u*l/muLiq
We = rhoLiq*u*u*l/sig
Bo = rhoLiq*g*l*l/sig
Fr = u/numpy.sqrt(g*l)

print(f"Bo = {Bo}")
print(f"Re = {Re}")
print(f"ins_invReynolds = {1/Re}")
print(f"ins_invWeber = {1/We}")
print(f"ins_gravY = {1/(Fr**2)}")
print(f"rhoGas = {rhoGas/rhoLiq}")
print(f"muGas = {muGas/muLiq}")
print(f"length = {l}")
print(f"velocity_ = {u}")
print(f"time = {l/u}")
