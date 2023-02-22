
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm

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
        self.weight_history = []
        self.weights = None
        

    def train(self, optimizer, lr_scheduler, iterations=10000):
        """
        Trains the neural network to solve the boundary value problem.

        Parametersones
        -----------
        optimizer: optimizer
            Optimizer to use for training
        lr_scheduler: lr_scheduler
            Learning rate scheduler to use for training
        iterations: int
            Number of iterations to train for
        """

        def adjust_weights(gradients):
            pde_gradient = tf.concat([tf.reshape(gradients[0][2*j], [-1]) for j in range(len(self.model.layers) - 1)], axis=0)
            data_gradient = tf.concat([tf.reshape(gradients[1][2*j], [-1]) for j in range(len(self.model.layers) - 1)], axis=0)
            new_weight = tf.reduce_max(tf.abs(pde_gradient)) / (tf.reduce_mean(tf.abs(data_gradient)))
            
            result = None
            for i, cond in enumerate(self.bvp.conditions):
                if cond.name != 'inner':
                    result = 0.5 * self.weights[i] + 0.5 * new_weight
                    self.weights[i].assign(result)
                    
            return result

        def compute_losses():
            criterion = tf.keras.losses.Huber()

            pdeloss = 0
            dataloss = 0
            for i, cond in enumerate(self.bvp.conditions):
                out = cond(self.model, self.bvp)
                if cond.name == 'inner':
                    pdeloss += self.weights[i] * criterion(out, 0)
                else:
                    dataloss += self.weights[i] * criterion(out, 0)
                    tf.print("Weight: ", self.weights[i])

            return pdeloss, dataloss

        def get_gradients():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.model.trainable_variables)
                pdeloss, dataloss = compute_losses()
                totalloss = pdeloss + dataloss

            pdegrad = tape.gradient(pdeloss, self.model.trainable_variables)
            datagrad = tape.gradient(dataloss, self.model.trainable_variables)
            totalgrad = tape.gradient(totalloss, self.model.trainable_variables)

            del tape

            return totalloss, (pdegrad, datagrad, totalgrad)

        @tf.function
        def train_step():
            if self.weights is None:
                self.weights = tf.Variable(np.ones(len(self.bvp.conditions)), dtype=tf.float32)
            
            # Compute current loss and gradient w.r.t. parameters
            loss, gradients = get_gradients()
            new_weight = adjust_weights(gradients)
            
            # Perform gradient descent step and update condition weights
            optimizer.apply_gradients(zip(gradients[2], self.model.trainable_variables))
            
            return loss, gradients, new_weight

        pbar = tqdm(range(iterations), desc='Pending...')
        for i in pbar:
            loss, gradients, new_weight = train_step()
            
            self.loss_history += [loss.numpy()]
            self.weight_history += [new_weight.numpy()]
            pbar.desc = 'loss = {:10.8e} lr = {:.5f}'.format(loss, lr_scheduler(i))

            if i % 1000 == 0:
                _, axs = plt.subplots(1, len(self.model.layers) - 1, figsize=(16, 4))
                for j in range(len(self.model.layers) - 1):
                    xs = np.linspace(-2, 2, 1000)
                    density = gaussian_kde(gradients[0][2*j].numpy().flatten())
                    axs[j].plot(xs, density(xs), 'orange', lw=0.5, label='PDE')

                    density = gaussian_kde(gradients[1][2*j].numpy().flatten())
                    axs[j].plot(xs, density(xs), 'b--', lw=0.5, label='Data')

                    axs[j].set_xlim(-2, 2)
                    axs[j].set_ylim(0, 100)
                    axs[j].set_yscale('symlog')
                    axs[j].set_title('Layer {}'.format(j + 1))
                    axs[j].legend()

                plt.show()
 
                plt.plot(self.weight_history)
                plt.xlabel('Update')
                plt.ylabel('Weight')
                plt.show() 