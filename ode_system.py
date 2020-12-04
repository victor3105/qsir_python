import numpy as np
import matplotlib.pyplot as plt
from my_solve_ivp import solve_ivp

N0 = 7300000.0
beta = 0.15
gamma = 0.013

t = np.linspace(0, 66, 66)


def dxdt_new(t, x, *args):
    N, beta, gamma = args
    S = -beta * x[0] * x[1] / N
    I = beta * x[0] * x[1] / N - gamma * x[1]
    R = gamma * x[1]
    return [S, I, R]

S0 = N0
I0 = 652
R0 = 13
x0 = S0, I0, R0

sol = solve_ivp(dxdt_new, (0, 66.0), [S0, I0, R0], method='RK45', args=(N0, beta, gamma), t_eval=t)
print(sol)

plt.plot(t, sol.y[0], 'r', label='S')
plt.plot(t, sol.y[1], 'g', label='I')
plt.plot(t, sol.y[2], 'b', label='R')
plt.legend(loc='best')
plt.grid()
plt.show()
