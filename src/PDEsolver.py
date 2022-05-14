import os
from time import time

from src.BoundaryValueProblem import *
from src.Visuals import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Solver:
    def __init__(self, bvp, num_inputs=2, num_outputs=1, num_hidden_layers=6, num_neurons_per_layer=16):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(num_inputs))

        for _ in range(num_hidden_layers):
            model.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                            activation=tf.keras.activations.tanh,
                                            kernel_initializer='glorot_normal'))

        model.add(tf.keras.layers.Dense(num_outputs))

        self.model = model
        self.bvp = bvp
        self.loss_history = []

    def train(self, optimizer, lr_scheduler, iterations=10000):
        def compute_loss():
            criterion = tf.keras.losses.Huber()

            loss = 0
            for cond in self.bvp.conditions:
                out = cond(self.model, self.bvp)
                loss += cond.weight * criterion(out, 0)

            return loss

        def get_grad():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.model.trainable_variables)
                loss = compute_loss()

            g = tape.gradient(loss, self.model.trainable_variables)
            del tape

            return loss, g

        @tf.function
        def train_step():
            # Compute current loss and gradient w.r.t. parameters
            loss, grad_theta = get_grad()

            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))

            return loss

        t0 = time()
        for i in range(iterations + 1):
            # while time() - t0 <= 60:
            loss = train_step()
            self.loss_history += [loss.numpy()]

            if i % 50 == 0:
                # if (time() - t0) % 5 < 0.01:
                # print('\rIt {:05d}s: loss = {:10.8e} lr = {:.5f}'.format(int(time() - t0), loss, 0), end="")
                print('\rIt {:05d}: loss = {:10.8e} lr = {:.5f}'.format(i, loss, lr_scheduler(i)), end="")

        print('\nComputation time: {:.1f} seconds'.format(time() - t0))


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
    plot_results()
