import numpy

rho_liq = 1000
rho_gas = 100
mu_liq = 10
mu_gas = 1
g = .98
sig = 24.5

r = 0.25
l = 2*r
u = numpy.sqrt(g*l)

Re_liq = rho_liq*u*l/mu_liq
Re_gas = rho_gas*u*l/mu_gas
We = rho_liq*u*u*l/sig
Bo = rho_liq*g*l*l/sig
Fr = u/numpy.sqrt(g*l)

print(f"Bo = {Bo}")
print(f"Re_liq = {Re_liq}")
print(f"Re_gas = {Re_gas}")
print(f"ins_invReynolds = {1/Re_liq}")
print(f"ins_invWeber = {1/We}")
print(f"ins_gravY = {1/(Fr**2)}")
print(f"rhoGas = {rho_gas/rho_liq}")
print(f"muGas = {mu_gas/mu_liq}")
print(f"length = {l}")
print(f"velocity_ = {u}")
print(f"time = {l/u}")

points = numpy.array([40, 80, 160, 320, 640])
h = 1./points/l

print(f"Points: {points}") 
print(f"Blocks: {points/8}")
print(f"Resolution: {h}")
print(f"Time-step: {h/50}")
print(f"Time_step: {h**2}")
