import os
from time import time
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Solver:
    def __init__(self, bvp, num_inputs=2, num_outputs=1, num_hidden_layers=6, num_neurons_per_layer=16):
        """
        Constructor for the Solver class.

        Parameters
        -----------
        bvp: BoundaryValueProblem
            Boundary value problem to be solved
        num_inputs: int
            Number of inputs to the neural network
        num_outputs: int
            Number of outputs from the neural network
        num_hidden_layers: int
            Number of hidden layers in the neural network
        num_neurons_per_layer:  int
            Number of neurons in each hidden layer
        """

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
        """
        Trains the neural network to solve the boundary value problem.

        Parameters
        -----------
        optimizer: optimizer
            Optimizer to use for training
        lr_scheduler: lr_scheduler
            Learning rate scheduler to use for training
        iterations: int
            Number of iterations to train for
        """

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
            loss = train_step()
            self.loss_history += [loss.numpy()]

            if i % 50 == 0:
                print('\rIt {:05d}: loss = {:10.8e} lr = {:.5f}'.format(i, loss, lr_scheduler(i)), end="")

        print('\nComputation time: {:.1f} seconds'.format(time() - t0))
