import numpy as np
import matplotlib.pyplot as plt
from my_solve_ivp import solve_ivp
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

from arizona_data import INFECTED, RECOVERED, DEAD, ALPHA, BETA, GAMMA


# Neural network for quarantine function
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(1))
predicted = np.ones((4, 66))


def my_loss(y_true, y_pred):
    infected = K.constant(INFECTED)
    recovered = K.constant(RECOVERED)
    dead = K.constant(DEAD)
    pred = K.constant(predicted)
    loss = K.sum((K.log(infected) - K.log(pred[1][:] + pred[3][:]))**2)
    loss += K.sum((K.log(recovered + dead) - K.log(pred[2][:]))**2)
    print(loss)
    return loss


model.compile(loss=my_loss, optimizer='adam', metrics=['accuracy'])


def train_ode(model, epochs, fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
              events=None, vectorized=False, args=None, **options):
    for i in range(epochs):
        sol = solve_ivp(fun, t_span, y0, method, t_eval, dense_output,
                        events, vectorized, args, **options)
        predicted[0][:] = sol.y[0]
        predicted[1][:] = sol.y[1]
        predicted[2][:] = sol.y[2]
        predicted[3][:] = sol.y[3]
        model.fit(np.array([[S0, I0, R0]]), np.array([1]), epochs=1)
        print(f'Epoch {i} out of {epochs}')
        print(model.get_weights())
        print()


N0 = 7300000.0
beta = 0.15
gamma = 0.013
delta = 0.01

t = np.linspace(0, 66, 66)


def dxdt_new(t, x, *args):
    N, beta, gamma, delta = args
    deltaInfected = beta * x[0] * x[1] / N
    quarantine = model.predict(np.expand_dims(x[:3], axis=0)) / N
    recoveredQ = delta * x[3]
    recoveredNoQ = gamma * x[1]
    S = -deltaInfected
    I = deltaInfected - recoveredNoQ - quarantine
    R = recoveredNoQ + recoveredQ
    Q = quarantine - recoveredQ
    return [S, I, R, Q]


S0 = N0
I0 = 652
R0 = 13
Q0 = 0
x0 = S0, I0, R0, Q0

train_ode(model, 100, fun=dxdt_new, t_span=(0, 66.0), y0=[S0, I0, R0, Q0], method='RK45', args=(N0, beta, gamma, delta), t_eval=t)

sol = solve_ivp(dxdt_new, (0, 66.0), [S0, I0, R0, Q0], method='RK45', args=(N0, beta, gamma, delta), t_eval=t)
# print(sol)
# predicted[1][:] = sol.y[1]
# predicted[2][:] = sol.y[2]

plt.plot(t, sol.y[0], 'r', label='S')
plt.plot(t, sol.y[1], 'g', label='I')
plt.plot(t, sol.y[2], 'b', label='R')
plt.plot(t, sol.y[3], 'k', label='Q')
plt.legend(loc='best')
plt.grid()
plt.show()
