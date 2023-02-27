import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
import logging

__all__ = ['Solver']	
logging.getLogger('tensorflow').setLevel(logging.ERROR)



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

        self.model = init_model(num_inputs, num_outputs, num_hidden_layers, num_neurons_per_layer)
        self.bvp = bvp
        self.loss_history = []
        self.weight_history = [1]
        self.weights = None
        self.step = None
        

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
            fig = plt.figure(layout='compressed', figsize=(16, 8))
            subfigs = fig.subfigures(2, 1, hspace=0)
            
            #self.model.summary()
            #print(' Layers: ', *[layer.name for layer in self.model.layers])
            #print('Gradients: ', *[gradient.shape for gradient in gradients[0]])

            axs = subfigs[0].subplots(1, len(self.model.layers) - 3)
            subfigs[0].suptitle('Gradient Distributions')
            for j in range(len(self.model.layers) - 1):
                j_mod = j
                if j in [1, 2]:
                    continue
                elif j > 2:
                    j_mod -= 2  

                xs = np.linspace(-2.5, 2.5, 1000)
                density = lambda i, x: gaussian_kde(gradients[i][2*j].numpy().flatten())(x)
                axs[j_mod].plot(xs, density(0, xs), 'orange', lw=0.5, label='PDE')
                axs[j_mod].plot(xs, density(1, xs), 'b--', lw=0.5, label='Data')

                axs[j_mod].set_xlim(min(xs), max(xs))
                axs[j_mod].set_ylim(0, 100)
                axs[j_mod].set_xscale('symlog')
                axs[j_mod].set_yscale('symlog')
                axs[j_mod].set_title(self.model.layers[j+1].name)
                axs[j_mod].legend()

            axs = subfigs[1].subplots(1, 2)
            axs[0].plot(self.loss_history, 'k', lw=0.5)
            axs[0].set_title('Loss History')
            axs[0].set_xlabel('Iteration')
            axs[0].set_ylabel('Loss')
            axs[0].set_yscale('log')

            axs[1].plot(self.weight_history, 'k', lw=0.5)
            axs[1].set_title('Weight History')
            axs[1].set_xlabel('#Update')
            axs[1].set_ylabel('Weight')

            figManager = plt.get_current_fig_manager()
            figManager.window.state("zoomed")
            plt.show()


        def adjust_weights(gradients):
            pde_gradient = tf.concat([tf.reshape(gradients[0][2*j], [-1]) for j in range(len(self.model.layers) - 1)], axis=0)
            data_gradient = tf.concat([tf.reshape(gradients[1][2*j], [-1]) for j in range(len(self.model.layers) - 1)], axis=0)
            new_weight = tf.reduce_max(tf.abs(pde_gradient)) / (tf.reduce_mean(tf.abs(data_gradient)))
            
            result = None
            for i, cond in enumerate(self.bvp.conditions):
                if cond.name != 'inner':
                    result = 0.25 * self.weights[i] + 0.75 * tf.maximum(1.0, new_weight)
                    self.weights[i].assign(result)
                    
            return result

        def compute_losses():
            criterion = tf.keras.losses.MeanSquaredError()

            pdeloss = 0
            dataloss = 0
            for i, cond in enumerate(self.bvp.conditions):
                out = cond(self.model, self.bvp)
                if cond.name == 'inner':
                    pdeloss += self.weights[i] * criterion(out, 0.0)
                else:
                    dataloss += self.weights[i] * criterion(out, 0.0)

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
                
            if self.step is None:
                self.step = tf.Variable(1)
            
            loss, gradients = get_gradients()
            
            if self.step % 50 == 0:
                new_weight = adjust_weights(gradients)
            else:
                new_weight = -1.0
                
            optimizer.apply_gradients(zip(gradients[2], self.model.trainable_variables))
            self.step.assign(self.step + 1)
            
            return loss, gradients, new_weight

        pbar = tqdm(range(iterations), desc='Pending...')
        for i in pbar:
            loss, gradients, new_weight = train_step()
            
            self.loss_history += [loss.numpy()]
            
            if new_weight != -1:
                self.weight_history += [new_weight.numpy()]
                
            pbar.desc = 'loss = {:10.8e} lr = {:.5f}'.format(loss, lr_scheduler(i))

            if i % debug_frequency == 0 or i == iterations - 1:
                debug(gradients)


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


def init_model(num_inputs, num_outputs, num_hidden_layers, num_neurons_per_layer):
    layer_sizes = [num_inputs] + [num_neurons_per_layer] * num_hidden_layers + [num_outputs] 
    layers = [Linear(layer_sizes[0], layer_sizes[1])] + \
             [ImprovedLinear(layer_sizes[i], layer_sizes[i + 1]) for i in range(1, len(layer_sizes) - 2)] + \
             [Linear(layer_sizes[-2], layer_sizes[-1])]

    encoder_1 = Encoder(num_inputs, num_neurons_per_layer)
    encoder_2 = Encoder(num_inputs, num_neurons_per_layer)

    inputs = tf.keras.Input(num_inputs)
    E1 = encoder_1(inputs)
    E2 = encoder_2(inputs)  
    for i in range(len(layers) - 1):
        if i == 0:
            outputs = layers[i](inputs)
        else:
            outputs = layers[i](outputs, E1, E2)

    outputs = layers[-1](outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    #model = tf.keras.Sequential()
    #model.add(tf.keras.Input(num_inputs))

    #for _ in range(num_hidden_layers):
        #model.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                        #activation=tf.keras.activations.tanh,
                                        #kernel_initializer='glorot_normal'))

    #model.add(tf.keras.layers.Dense(num_outputs))

    return model


if __name__ == '__main__':
    from Sampling import Cuboid, Equidistant
    txs = Cuboid([0, 0], [1, 1]).pick(9, Equidistant())

    model = init_model(2, 1, 4, 50)
    print(model(txs))