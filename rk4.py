import numpy as np
import matplotlib.pyplot as plt

# define constants
g_l = 2.0
g_ca = 4.0
g_k = 8.0
e_l = -50.0
e_ca = 100.0
e_k = -70.0
tau = 0.1 
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

# define functions for RK method
def f(v, n, i):
  return (i - g_l*(v-e_l) - g_ca*m_ss(v)*(v-e_ca) - g_k*n*(v-e_k)) / cap

def g(v, n, i):
  return (n_ss(v) - n) / tau_n(n)

def k1(v, n, i, h):
  return h*f(v, n, i)
def j1(v, n, i, h):
  return h*g(v, n, i)

def k2(v, n, i, h):
  return h*f(v + k1(v, n, i, h)/2, n + j1(v, n, i, h)/2, i)
def j2(v, n, i, h):
  return h*g(v + k1(v, n, i, h)/2, n + j1(v, n, i, h)/2, i)

def k3(v, n, i, h):
  return h*f(v + k2(v, n, i, h)/2, n + j2(v, n, i, h)/2, i)
def j3(v, n, i, h):
  return h*g(v + k2(v, n, i, h)/2, n + j2(v, n, i, h)/2, i)

def k4(v, n, i, h):
  return h*f(v + k3(v, n, i, h), n + j3(v, n, i, h), i)
def j4(v, n, i, h):
  return h*g(v + k3(v, n, i, h), n + j3(v, n, i, h), i)

def v_next(v, n, h, i):
  return v + (1.0/6.0)*(k1(v, n, i, h)+2*k2(v, n, i, h)+2*k3(v, n, i, h)+k4(v, n, i, h))

def n_next(v, n, h):
  return n + (1.0/6.0)*(j1(v, n, i, h)+2*j2(v, n, i, h)+2*j3(v, n, i, h)+j4(v, n, i, h))

# calculate points
x_points = np.linspace(-50.0, 50.0, 1000)
vpoints = vnullvec(x_points, 300.0)
npoints = nnullvec(x_points)

t_points = np.arange(0, 1999, 1)
vout = np.zeros(2000) # initial conditions
nout = np.zeros(2000)
vout[0] = -55.0
nout[0] = n_ss(-55.0)
for i in t_points:
  vout[i+1] = v_next(vout[i], nout[i], 0.2, 300.0)
  nout[i+1] = n_next(vout[i], nout[i], 0.2)

# plot
plt.subplot(121)
plt.plot(vout)
plt.axis([0, 800, -60, 60])
plt.xlabel('t, ms')
plt.ylabel('V, mV')
plt.title('Voltage over Time')

plt.subplot(122)
plt.plot(x_points, vpoints, label="V nullcline")
plt.plot(x_points, npoints, label="n nullcline")
plt.plot(vout, nout, label="V-n phase plot")
plt.axis([-50, 50, 0, 1])
plt.xlabel('V, mV')
plt.ylabel('n')
plt.title('Nullclines')
plt.legend()
plt.show()

