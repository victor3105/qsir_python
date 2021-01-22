import pygad
import pygad.kerasga

import numpy as np
import matplotlib.pyplot as plt
from my_solve_ivp import solve_ivp
from keras.models import Sequential
from keras.layers import Dense

from arizona_data import INFECTED, RECOVERED, DEAD, ALPHA, BETA, GAMMA

# Neural network for quarantine function
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(1))
# contains S, I, R, Q values in rows
predicted = np.ones((4, 66))


def my_loss(y_true, y_pred):
    infected = INFECTED
    recovered = RECOVERED
    dead = DEAD
    pred = predicted
    # get rid of negative values
    pred = np.where(pred < 0, 1e9, pred)
    loss = np.sum((np.log(infected) - np.log((pred[1][:] + pred[3][:])))**2)
    loss += np.sum((np.log(recovered + dead) - np.log((pred[2][:])))**2)
    print(f'Loss: {loss}')
    print()
    return loss


def fitness_func(solution, sol_idx):
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)

    t = np.linspace(0, 66, 66)
    sol = solve_ivp(dxdt_new, (0, 66.0), [S0, I0, R0, Q0], method='RK45', args=(N0, beta, gamma, delta), t_eval=t)
    predicted[0][:] = sol.y[0]
    predicted[1][:] = sol.y[1]
    predicted[2][:] = sol.y[2]
    predicted[3][:] = sol.y[3]
    loss = my_loss(1, 1)
    solution_fitness = 1.0 / loss

    return solution_fitness


def train_ode(model, epochs, fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
              events=None, vectorized=False, args=None, **options):
    for i in range(epochs):
        sol = solve_ivp(fun, t_span, y0, method, t_eval, dense_output,
                        events, vectorized, args, **options)
        predicted[0][:] = sol.y[0]
        predicted[1][:] = sol.y[1]
        predicted[2][:] = sol.y[2]
        predicted[3][:] = sol.y[3]
        # model.fit(np.array([[S0, I0, R0]]), np.array([1]), epochs=1)
        # print(f'Epoch {i + 1} out of {epochs}')
        # print(model.get_weights())
        # print()


# create random array with quarantine values and use standard loss function
N0 = 7300000.0
beta = 0.15
gamma = 0.013
delta = 0.01


def dxdt_new(t, x, *args):
    N, beta, gamma, delta = args
    deltaInfected = beta * x[0] * x[1] / N
    quarantine = model.predict(np.expand_dims(x[:3], axis=0))[0][0] * x[1] / (N * 100)
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

weights_vector = pygad.kerasga.model_weights_as_vector(model=model)
keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=30)
num_generations = 200
num_parents_mating = 5
initial_population = keras_ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func)
ga_instance.run()
ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

t = np.linspace(0, 66, 66)
sol = solve_ivp(dxdt_new, (0, 66.0), [S0, I0, R0, Q0], method='RK45', args=(N0, beta, gamma, delta), t_eval=t)

plt.plot(t, sol.y[0], 'r', label='S')
plt.plot(t, sol.y[1], 'g', label='I')
plt.plot(t, sol.y[2], 'b', label='R')
plt.plot(t, sol.y[3], 'k', label='Q')
plt.legend(loc='best')
plt.grid()
plt.show()

plt.plot(t, sol.y[1], 'r', label='predicted')
plt.plot(t, INFECTED, 'g', label='real')
plt.legend(loc='best')
plt.grid()
plt.show()
