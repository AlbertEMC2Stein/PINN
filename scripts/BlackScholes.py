from PDESolver import *
from scipy.stats import norm
import matplotlib.pyplot as plt

class BlackScholes(BoundaryValueProblem):
    def __init__(self, strike=5.0, maturity=1.0, volatility=0.45, interest_rate=0.03):
        super().__init__()

        self.conditions = [
            Condition(
                "final",
                lambda Du: Du["C"] - tf.maximum(Du["S"] - strike, 0),
                (Cuboid([maturity, 0], [maturity, 10]), 1024)
            ),
            Condition(
                "inner",
                lambda Du: 0.5 * volatility ** 2 * Du["S"] ** 2 * Du["C_SS"] + interest_rate * Du["S"] * Du["C_S"] + Du["C_t"] - interest_rate * Du["C"],
                (Cuboid([0, 0], [maturity, 10]), 1024)
            )
        ]

        self.specification = Specification(["C"], ["t", "S"], ["C_tt", "C_SS"])

# Number of iterations
N = 20000

# Initialize solver, learning rate scheduler and choose optimizer
optim = Optimizer(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.95)
solver = Solver(BlackScholes(), optim, num_hidden_layers=5, num_neurons_per_layer=20)

# Train model and plot results
solver.train(iterations=N, debug_frequency=N)

def fair_call_price(S, strike=5.0, maturity=1.0, volatility=0.45, interest_rate=0.03):
    d1 = lambda S: (np.log(S / strike) + (interest_rate + 0.5 * volatility ** 2) * maturity) / (volatility * np.sqrt(maturity))
    d2 = lambda S: d1(S) - volatility * np.sqrt(maturity)
    return S * norm.cdf(d1(S)) - strike * np.exp(-interest_rate * maturity) * norm.cdf(d2(S))

ts = np.zeros(100)
Ss = np.linspace(0.1, 10, 100)
plt.plot(Ss, fair_call_price(Ss), 'b-', label="Analytical")
plt.plot(Ss, solver.evaluate({"t": list(ts), "S": list(Ss)})[:, 0], 'r--', label="Numerical")
plt.xlabel("Current Stock price")
plt.ylabel("Call price")
plt.legend()
plt.show()

