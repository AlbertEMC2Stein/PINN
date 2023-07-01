from PDESolver import *
from scipy.integrate import quad_vec
import matplotlib.pyplot as plt

class BlackScholes(BoundaryValueProblem):
    def __init__(self, final_price_fun, strike=5.0, maturity=1.0, volatility=0.45, interest_rate=0.03, formulation='REGULAR'):
        super().__init__()

        if formulation == 'REGULAR':
            self.conditions = [
                Condition(
                    "initial",
                    lambda Du: Du["C"],
                    (Cuboid([0, 0], [maturity, 0]), 128)
                ),
                Condition(
                    "final",
                    lambda Du: Du["C"] - final_price_fun(Du["S"]),
                    (Cuboid([maturity, 0], [maturity, 20]), 128)
                ),
                Condition(
                    "inner",
                    lambda Du: 0.5 * volatility ** 2 * Du["S"] ** 2 * Du["C_SS"] + interest_rate * Du["S"] * Du["C_S"] + Du["C_t"] - interest_rate * Du["C"],
                    (Cuboid([0, 0], [maturity, 20]), 128)
                )
            ]

            self.specification = Specification(["C"], ["t", "S"], ["C_t", "C_SS"])
            
        # https://www.math.tamu.edu/~stecher/425/Sp12/blackScholesHeatEquation.pdf
        elif formulation == 'HEAT':
            alpha = (volatility ** 2 - 2 * interest_rate) / (2 * volatility ** 2)

            self.conditions = [
                Condition(
                    "initial",
                    lambda Du: Du["u"] - tf.exp(-alpha * Du["x"]) * final_price_fun(tf.exp(Du["x"])),
                    (Cuboid([0, 0], [0, 3]), 256)
                ),
                Condition(
                    "inner",
                    lambda Du: Du["u_t"] - Du["u_xx"],
                    (Cuboid([0, 0], [maturity * volatility ** 2 / 2, 3]), 256)
                )
                # 1 < S < 20 
            ]

            self.specification = Specification(["u"], ["t", "x"], ["u_t", "u_xx"])

def fair_call_price(S, final_price_fun, maturity, volatility, interest_rate):
    alpha = (volatility ** 2 - 2 * interest_rate) / (2 * volatility ** 2)
    beta = -((volatility ** 2 + 2 * interest_rate) / (2 * volatility ** 2))**2

    integrand = lambda x: np.exp(-alpha * x) * final_price_fun(np.exp(x)) * np.exp(- (np.log(S) - x) ** 2 / (2 * volatility ** 2 * maturity))
    u = 1 / np.sqrt(2 * np.pi * volatility ** 2 * maturity) * quad_vec(integrand, -20, 20)[0]

    return np.exp(alpha * np.log(S) + beta * (volatility ** 2 * maturity / 2)) * u


def heat_to_regular(solver, S, maturity, volatility, interest_rate):
    alpha = (volatility ** 2 - 2 * interest_rate) / (2 * volatility ** 2)
    beta = -((volatility ** 2 + 2 * interest_rate) / (2 * volatility ** 2))**2

    ts = list(np.ones(len(S)) * maturity * volatility ** 2 / 2)
    xs = list(np.ones(len(S)) * np.log(S))

    us = solver.evaluate({"t": ts, "x": xs})[:, 0]

    return np.exp(alpha * np.log(S) + beta * (volatility ** 2 * maturity / 2)) * us

# Number of iterations
N = 15000
strike = 5.0
maturity = 1.0
volatility = 0.3
interest_rate = 0.01
formulation = 'HEAT'

f = lambda x: tf.maximum(strike - 2.0 * tf.abs(x - strike), 0)#tf.maximum(x - strike, 0) #- tf.maximum(x - strike - 5, 0)

# Initialize solver, learning rate scheduler and choose optimizer
optim = Optimizer(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9)
solver = Solver(BlackScholes(f, strike, maturity, volatility, interest_rate, formulation),
                optim, num_hidden_layers=5, num_neurons_per_layer=20, activation=tf.keras.activations.tanh)

# Train model and plot results
solver.train(iterations=N, debug_frequency=-1)

Ss = np.linspace(1, 20, 100)
if formulation == 'REGULAR':
    ts = np.zeros(len(Ss))
    Cs = solver.evaluate({"t": list(ts), "S": list(Ss)})[:, 0]
elif formulation == 'HEAT':
    Cs = heat_to_regular(solver, Ss, maturity, volatility, interest_rate)

plt.plot(Ss, fair_call_price(Ss, f, maturity, volatility, interest_rate), 'b-', label="Analytical")
plt.plot(Ss, Cs, 'r--', label="Numerical")
plt.xlabel("Current Stock price")
plt.ylabel("Call price")
plt.legend()
plt.show()

file_name = f'BS_{strike}_{maturity}_{volatility}_{interest_rate}.csv'
file_name = file_name.replace('.', 'd')
np.savetxt(f'Results/{file_name}', np.column_stack((Ss, Cs)), delimiter=',', header='S, C', comments='', fmt='%.5f')

