import numpy as np
import matplotlib.pyplot as plt

# define constants
g_l = 2.0
g_ca = 4.0
g_k = 8.0
e_l = -50.0
e_ca = 100.0
e_k = -70.0
tau = 10.0
v1 = -1.2 
v2 = 18.0
v3 = 2.0 
v4 = 30.0 
cap = 20.0

# steady state functions
def m_ss(v):
  return (1.0/2.0) * (1.0 + np.tanh((v - v1)/v2))

def n_ss(v):
  return (1.0/2.0) * (1.0 + np.tanh((v - v3)/v4))

def tau_n(v):
  return 1.0 / (tau * np.cosh((v - v3)/(2*v4)))

# define nullcline functions then vectorize them
def vnull(v, i):
  return (i - g_l*(v-e_l) - g_ca*m_ss(v)*(v-e_ca)) / (g_k*(v-e_k))
vnullvec = np.vectorize(vnull)

def nnull(v):
  return n_ss(v)
nnullvec = np.vectorize(nnull)

# calculate points
x_points = np.linspace(-50.0, 50.0, 1000)
vpoints = vnullvec(x_points, 300.0)
npoints = nnullvec(x_points)

# plot
plt.plot(x_points, vpoints, label="V nullcline")
plt.plot(x_points, npoints, label="n nullcline")
plt.axis([-50, 50, 0, 1])
plt.xlabel('V, mV')
plt.ylabel('n')
plt.title('Nullclines')
plt.legend()
plt.show()
