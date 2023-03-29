import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
import warnings
import logging
import os
from sys import stdout

__all__ = ['Solver']	

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Solver:
    def __init__(self, bvp, num_hidden_layers=4, num_neurons_per_layer=50):
        """
        Constructor for the Solver class.

        Parameters
        -----------
        bvp: BoundaryValueProblem
            Boundary value problem to be solved
        num_hidden_layers: int
            Number of hidden layers in the neural network
            Defaults to 4
        num_neurons_per_layer:  int
            Number of neurons in each hidden layer
            Defaults to 50
        """

        num_inputs = len(bvp.get_specification()['variables'])
        num_outputs = len(bvp.get_specification()['components'])
        inner_constraint = [condition for condition in bvp.get_conditions() if condition.name == 'inner'][0]
        mean, variance = inner_constraint.get_normalization_constants()

        self.model = init_model(num_inputs, num_outputs, num_hidden_layers, num_neurons_per_layer, mean, variance)
        self.bvp = bvp

        self.loss_history = []
        self.weight_history = [1]
        self.weights = None
        self.step = None
        self.sample_history = []

    def train(self, optimizer, lr_scheduler, iterations=10000, debug_frequency=2500):
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

        def debug(gradients):
            fig = plt.figure(figsize=(16, 8), layout='compressed')
            subfigs = fig.subfigures(2, 1, hspace=0)

            axs = subfigs[0].subplots(1, len(self.model.layers) - 4)
            subfigs[0].suptitle('Gradient Distributions')

            ax_count = -1
            for j in range(len(self.model.layers)):
                if j in [0, 1, 3, 4]:
                    continue
                else:
                    ax_count += 1

                layer_gradient = lambda i: tf.concat([tf.reshape(gradient, [-1]) for gradient in gradients[i][2*(j-2):2*(j-1)]], axis=0)
                density = lambda i, x: gaussian_kde(layer_gradient(i).numpy())(x)

                xs = np.linspace(-2.5, 2.5, 1000)
                axs[ax_count].plot(xs, density(0, xs), 'orange', lw=0.5, label='PDE')
                axs[ax_count].plot(xs, density(1, xs), 'b--', lw=0.5, label='Data')

                axs[ax_count].set_xlim(min(xs), max(xs))
                axs[ax_count].set_ylim(0, 100)
                axs[ax_count].set_yscale('symlog')
                axs[ax_count].set_title(self.model.layers[j].name)
                axs[ax_count].legend()

            n = len(self.loss_history)
            k = min(100, n)
            averaged_loss = np.convolve(self.loss_history, np.ones(k) / k, mode='same')

            axs = subfigs[1].subplots(1, 2)
            axs[0].semilogy(range(n), self.loss_history, 'k-', lw=0.5)
            axs[0].semilogy(range(n), averaged_loss, 'r--', lw=1)
            axs[0].set_title('Loss History')
            axs[0].set_xlabel('Iteration')
            axs[0].set_ylabel('Loss')
            axs[0].set_xlim(0, n - (1 if n > 1 else 0))

            axs[1].plot(self.weight_history, 'k', lw=0.5)
            axs[1].set_title('Weight History')
            axs[1].set_xlabel('#Update')
            axs[1].set_ylabel('Weight')
            axs[1].set_xlim(0, len(self.weight_history) - 1)

            figManager = plt.get_current_fig_manager()
            figManager.window.state("zoomed")
            
            plt.show()

        def adjust_weights(gradients):
            pde_gradient = tf.concat([tf.reshape(gradient, [-1]) for gradient in gradients[0]], axis=0)
            data_gradient = tf.concat([tf.reshape(gradient, [-1]) for gradient in gradients[1]], axis=0)
            new_weight = tf.reduce_max(tf.abs(pde_gradient)) / (tf.reduce_mean(tf.abs(data_gradient)))
            
            result = None
            for i, cond in enumerate(self.bvp.get_conditions()):
                if cond.name != 'inner':
                    result = 0.25 * self.weights[i] + 0.75 * new_weight
                    self.weights[i].assign(result)
                    
            return result

        def compute_losses():
            criterion = tf.keras.losses.MeanSquaredError()

            pdeloss = 0
            dataloss = 0
            for i, cond in enumerate(self.bvp.get_conditions()):
                out, samples = cond(self.model, self.bvp)
                if cond.name == 'inner':
                    pdeloss += self.weights[i] * criterion(out, 0.0)
                else:
                    dataloss += self.weights[i] * criterion(out, 0.0)

            return pdeloss, dataloss, samples 

        def get_gradients():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.model.trainable_variables)
                pdeloss, dataloss, samples = compute_losses()
                totalloss = pdeloss + dataloss

                pdegrad = tape.gradient(pdeloss, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                datagrad = tape.gradient(dataloss, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                totalgrad = tape.gradient(totalloss, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            del tape

            return totalloss, (pdegrad, datagrad, totalgrad), samples

        @tf.function
        def train_step():
            if self.weights is None:
                self.weights = tf.Variable(np.ones(len(self.bvp.get_conditions())), dtype=tf.float32)
                
            if self.step is None:
                self.step = tf.Variable(0)
            
            loss, gradients, samples = get_gradients()
            
            if self.step % 10 == 0:
                new_weight = adjust_weights(gradients)
            else:
                new_weight = -1.0
                
            optimizer.apply_gradients(zip(gradients[2], self.model.trainable_variables))
            self.step.assign(self.step + 1)
            
            return loss, gradients, new_weight, samples

        pbar = tqdm(range(iterations), desc='Pending...')
        for i in pbar:
            loss, gradients, new_weight, samples = train_step()
            
            self.loss_history += [loss.numpy()]
            self.sample_history = self.sample_history[-99:] + [samples]
            
            if new_weight != -1:
                self.weight_history += [new_weight.numpy()]
            
            avgloss = np.mean(self.loss_history[-100:])
            pbar.desc = 'øloss = {:10.8e} lr = {:.5f}'.format(avgloss, lr_scheduler(i))

            if i % debug_frequency == 0 or i == iterations - 1:
                debug(gradients)
                
                #n = min(i, 100)
                #color = plt.cm.rainbow(np.linspace(0, 1, n))
                #for j in range(n):
                #    plt.scatter(*self.sample_history[j].numpy().T, s=0.5, c=[color[j]] * len(self.sample_history[j]), cmap='viridis')
                
                #plt.title("Samples generated in the last 100 iterations")
                #plt.show()


###################################################################################
###################################################################################
###################################################################################

                
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
    return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                        dtype=tf.float32, trainable=True)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, inputs, outputs, activation=tf.tanh):
        super(Encoder, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation

    def build(self, input_shape):  
        self.W = xavier_init([self.inputs, self.outputs]) 
        self.b = xavier_init([1, self.outputs])

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return self.activation(tf.add(tf.matmul(inputs, self.W), self.b))
    

class ImprovedLinear(tf.keras.layers.Layer):
    def __init__(self, inputs, outputs, activation=tf.tanh):
        super(ImprovedLinear, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation

    def build(self, input_shape):  
        self.W = xavier_init([self.inputs, self.outputs]) 
        self.b = xavier_init([1, self.outputs])

    def call(self, inputs, encoder_1, encoder_2):  
        return tf.math.multiply(self.activation(tf.add(tf.matmul(inputs, self.W), self.b)), encoder_1) + \
                tf.math.multiply(1 - self.activation(tf.add(tf.matmul(inputs, self.W), self.b)), encoder_2)


class Linear(tf.keras.layers.Layer):
    def __init__(self, inputs, outputs):
        super(Linear, self).__init__()
        self.inputs = inputs
        self.outputs = outputs

    def build(self, input_shape):  
        self.W = xavier_init([self.inputs, self.outputs]) 
        self.b = xavier_init([1, self.outputs])

    def call(self, inputs):  
        return tf.add(tf.matmul(inputs, self.W), self.b)


def init_model(num_inputs, num_outputs, num_hidden_layers, num_neurons_per_layer, mean, variance):
    layer_sizes = [num_inputs] + [num_neurons_per_layer] * num_hidden_layers + [num_outputs] 
    layers = [Linear(layer_sizes[0], layer_sizes[1])] + \
             [ImprovedLinear(layer_sizes[i], layer_sizes[i + 1]) for i in range(1, len(layer_sizes) - 2)] + \
             [Linear(layer_sizes[-2], layer_sizes[-1])]

    encoder_1 = Encoder(num_inputs, num_neurons_per_layer)
    encoder_2 = Encoder(num_inputs, num_neurons_per_layer)

    inputs = tf.keras.Input(num_inputs)
    outputs = tf.keras.layers.Normalization(axis=-1, mean=mean, variance=variance)(inputs)

    E1 = encoder_1(outputs)
    E2 = encoder_2(outputs)  
    for i in range(len(layers) - 1):
        if i == 0:
            outputs = layers[i](outputs)
        else:
            outputs = layers[i](outputs, E1, E2)

    outputs = layers[-1](outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # model = tf.keras.Sequential()
    # model.add(tf.keras.Input(num_inputs))

    # for _ in range(num_hidden_layers):
    #     model.add(tf.keras.layers.Dense(num_neurons_per_layer,
    #                                     activation=tf.keras.activations.tanh,
    #                                     kernel_initializer='glorot_normal'))

    # model.add(tf.keras.layers.Dense(num_outputs))

    return model


if __name__ == '__main__':
    from Sampling import Cuboid, Equidistant

    tf.random.set_seed(1)
    txs = Cuboid([0, 0], [1, 1]).pick(9, Equidistant())

    model = init_model(2, 1, 4, 50, mean=-1, variance=1)

    model.summary()
    print(model(txs)) 