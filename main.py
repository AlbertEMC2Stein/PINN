from PDESolver.Solver import *
from PDESolver.Visuals import *
from PDESolver.BoundaryValueProblem import *


if __name__ == "__main__":
    bvp = ControlledHeatEquation1D

    # Number of iterations
    N = 5000

    # Initialize model, learning rate scheduler and choose optimizer
    solver = Solver(bvp, num_inputs=2, num_outputs=2)
    lr = tf.keras.optimizers.schedules.PolynomialDecay(0.1, N, 1e-5, 2)
    optim = tf.keras.optimizers.Adam(learning_rate=lr)

    #tf.Variable(1., trainable=True, name="lambda")
    solver.train(optim, lr, N)

    #plot_phaseplot()
    #plot_2D()
    #plot_2Dvectorfield()
    plot_1D(solver)