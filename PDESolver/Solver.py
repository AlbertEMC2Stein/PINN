import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy.stats import gaussian_kde
from tqdm import tqdm
import warnings
import logging
import os

__all__ = ['Solver', 'Optimizer']	

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Solver:
    def __init__(self, bvp, optimizer, num_hidden_layers=4, num_neurons_per_layer=50):
        """
        Constructor for the Solver class.

        Parameters
        -----------
        bvp: BoundaryValueProblem
            Boundary value problem to be solved
        optimizer: Optimizer
            Optimizer to be used for training the neural network
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
        self.optimizer = optimizer

        self.loss_history = []
        self.weight_history = [{cond.name: 1. for cond in bvp.get_conditions()}]
        self.weights = None
        self.step = None

    def compute_differentials(self, samplePoints):
        """
        Calculates the differentials of the model at the given points.

        Parameters
        -----------
        model: neural network
            Model to calculate the differentials of
        samplePoints: tensor
            Tensor of points to calculate the differentials at

        Returns
        -----------
        gradient_dict: dict
            Dictionary containing the differentials of the model needed for the bvp at the given points
        """

        specs = self.bvp.get_specification()

        stack = lambda *tensors: tf.concat(tensors, axis=1)
        component_funs = {component: lambda x, index=i: self.model(x)[:, index] 
                          for i, component in enumerate(specs["components"])}

        gradient_dict = {}

        with tf.GradientTape(persistent=True) as tape:
            for i, variable in enumerate(specs["variables"]):
                gradient_dict[variable] = samplePoints[:, i:i + 1]
                tape.watch(gradient_dict[variable])

            watched = [gradient_dict[variable] for variable in specs["variables"]]
            for component in specs["components"]:
                output = component_funs[component](tf.stack(watched, axis=1))
                gradient_dict[component] = tf.reshape(output, (-1, 1))
    
            for differential in specs["differentials"]:
                component = differential.split("_")[0]
                variables = differential.split("_")[1]

                differential_head = component 
                for i, variable in enumerate(variables):
                    if i == 0:
                        differential_head_new = differential_head + "_" + variable
                    else:
                        differential_head_new = differential_head + variable

                    if not differential_head_new in gradient_dict:
                        gradient_dict[differential_head_new] = tape.gradient(gradient_dict[differential_head], gradient_dict[variable])
                        differential_head = differential_head_new

        for (stacked_component_name, components) in specs["stacked_components"].items():
            stacked = stack(*[gradient_dict[component] for component in components])
            gradient_dict[stacked_component_name] = stacked

        return gradient_dict
   
    def compute_residuals(self):
        """
        Gets the residuals for the boundary value problem.

        Returns
        -----------
        dict: Dictionary of residuals of form {condition_name: residual}
        """

        conditions = self.bvp.get_conditions()

        residuals = {}
        for condition in conditions:
            samples = condition.sample_points()
            Du = self.compute_differentials(samples)
            residuals[condition.name] = condition.residue_fn(Du)

        return residuals
  
    def compute_losses(self):
        """
        Computes the losses for the neural network.
        
        Returns
        -----------
        tuple: Tuple of losses for the PDE and data
        """

        criterion = tf.keras.losses.MeanSquaredError()
        residuals = self.compute_residuals()

        losses = {name: 0.0 for name in residuals.keys()}
        losses['L2'] = 0.0
        for i, (name, residual) in enumerate(residuals.items()):
            losses[name] += self.weights[i] * criterion(residual, 0.0)
            losses['L2'] += criterion(residual, 0.0)

        losses['total'] = sum(losses.values())

        return losses
 
    def compute_gradients(self):
        """
        Gets the gradients of the neural network.

        Returns
        -----------
        tuple: Tuple of gradients
        """
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            losses = self.compute_losses()

            kwargs = {'sources': self.model.trainable_variables, 'unconnected_gradients': tf.UnconnectedGradients.ZERO}
            grads = {name: tape.gradient(loss, **kwargs) for name, loss in losses.items() if name != 'L2'}

        return losses['L2'], grads


    def adjust_weights(self, gradients):
        """
        Adjusts the weights of the PDE and data losses.

        Parameters
        -----------
        gradients: tuple
            Tuple of gradients of the PDE and data losses obtained from compute_gradients
        """

        gradient_vectors = [tf.concat([tf.reshape(gradient, [-1]) for gradient in gradients_list], axis=0) for gradients_list in gradients.values()][:-1] # excluding 'totalgrad'
        variances = [tf.math.reduce_variance(gradient) for gradient in gradient_vectors]
        most_varying = tf.math.argmax(variances)
        most_varying_absmax = tf.math.reduce_max(tf.abs(tf.gather(gradient_vectors, most_varying)))

        new_weights = {}
        minimal = tf.float32.max
        for i, cond in enumerate(self.bvp.get_conditions()):
            name = cond.name
            if i != most_varying:
                new_weight = most_varying_absmax / (tf.reduce_mean(tf.abs(gradient_vectors[i])))
                new_weight = 0.75 * self.weights[i] + 0.25 * new_weight

                new_weights[name] = new_weight
                self.weights[i].assign(new_weight)
            else:
                new_weights[name] = self.weights[i]

            minimal = tf.math.minimum(minimal, new_weights[name])

        new_weights = {name: new_weights[name] / minimal for name in new_weights.keys()}
                
        return new_weights
    
    @tf.function
    def train_step(self):
        """
        Performs a single training step.

        Parameters
        -----------
        optimizer: optimizer
            Optimizer to use for training
        """

        if self.step is None:
            self.step = tf.Variable(0)
            self.weights = tf.Variable(np.ones(len(self.bvp.get_conditions())), dtype=tf.float32)
        
        l2loss, gradients = self.compute_gradients()
        
        if self.step % 10 == 0:
            new_weights = self.adjust_weights(gradients)  
        else:
            new_weights = {cond.name: -1. for cond in self.bvp.get_conditions()}          
            
        self.optimizer.apply_gradients(zip(gradients['total'], self.model.trainable_variables))
        self.step.assign(self.step + 1)
        
        return l2loss, gradients, new_weights

    def train(self, iterations=10000, debug_frequency=2500):
        """
        Trains the neural network to solve the boundary value problem.

        Parameters
        -----------
        iterations: int (default=10000)
            Number of iterations to train for
        debug_frequency: int (default=2500)
            Frequency (every X iterations) at which to show debug panel.
            If negative, no debug panel is shown
        """

        best_loss = np.inf
        iterations_since_last_improvement = 0
        k_max = int(np.ceil(np.log10(iterations))) + 1
        pbar = tqdm(range(iterations), desc='Pending...')

        for i in pbar:
            l2loss, gradients, new_weights = self.train_step()
            
            self.loss_history += [l2loss.numpy()]
            
            if l2loss.numpy() < best_loss:
                best_loss = l2loss.numpy()
                iterations_since_last_improvement = 0
            else:
                iterations_since_last_improvement += 1
            
            if list(new_weights.values())[0] != -1:
                self.weight_history += [{name: new_weights[name].numpy() for name in new_weights.keys()}]
            
            avg_loss = np.mean(self.loss_history[-100:])
            pbar.desc = f'øL²-loss = {avg_loss:.3e} (best: {best_loss:.3e}, {iterations_since_last_improvement:0{k_max}d}it ago) lr = {self.optimizer.lr.numpy():.5f}'

            if debug_frequency > 0 and (i % debug_frequency == 0 or i == iterations - 1):
                self.show_debugplot(gradients)

    def show_debugplot(self, gradients):
        """
        Shows a debug plot of the neural network.

        Parameters
        -----------
        gradients: tuple
            Tuple of gradients of the PDE and data losses obtained from compute_gradients
        """

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

            layer_gradient = lambda name: tf.concat([tf.reshape(gradient, [-1]) for gradient in gradients[name][2*(j-2):2*(j-1)]], axis=0)
            density = lambda i, x: gaussian_kde(layer_gradient(i).numpy())(x)

            xs = np.linspace(-2.5, 2.5, 1000)
            for cond in self.bvp.get_conditions():
                axs[ax_count].plot(xs, density(cond.name, xs), lw=0.5, label=cond.name)

            axs[ax_count].set_xlim(min(xs), max(xs))
            axs[ax_count].set_ylim(0, 100)
            axs[ax_count].set_yscale('symlog')
            axs[ax_count].set_title(self.model.layers[j].name)
        
        axs[-1].legend()

        n = len(self.loss_history)
        averaged_loss = [np.mean(self.loss_history[:k+1] if k < 99 else self.loss_history[k-99:k+1]) for k in range(n)]
        best_loss = np.min(self.loss_history)

        axs = subfigs[1].subplots(1, 2)
        trans = blended_transform_factory(axs[0].transAxes, axs[0].transData)
        loss_handle, = axs[0].semilogy(range(n), self.loss_history, 'k-', lw=0.5, alpha=0.5, label='Loss')
        avg_loss_handle, = axs[0].semilogy(range(n), averaged_loss, 'r--', lw=1, label=f'$ø_{{{min(100, n)}}}$ Loss')
        axs[0].axhline(best_loss, color='g', lw=0.5)
        axs[0].text(0.995, 0.97 * best_loss, f'Best: {best_loss:.3e}', ha='right', va='top', color='g', transform=trans)

        ax = axs[0].twinx()
        ax.set_yscale('log')
        lr_handle, = ax.plot(self.optimizer.lr_history_upto(n), 'b-', lw=0.5, label='Learning rate')

        axs[0].set_title('Loss (left) and learning rate (right) history')
        axs[0].set_xlabel('Iteration')
        axs[0].set_xlim(0, n - (1 if n > 1 else 0))

        if n > 1:
            handles = [loss_handle, avg_loss_handle, lr_handle]
            axs[0].legend(handles=handles, loc='lower left')

        for cond in self.bvp.get_conditions():
            cond_weight_history = [self.weight_history[k][cond.name] for k in range(len(self.weight_history))]
            axs[1].plot(cond_weight_history, lw=0.5)

        axs[1].set_title('Weight History')
        axs[1].set_xlabel('Update #')
        axs[1].set_xlim(0, len(self.weight_history) - 1)
        axs[1].set_yscale('log')

        if os.name == 'nt':
            figManager = plt.get_current_fig_manager()
            figManager.window.state("zoomed")
        
        plt.show()


class Optimizer(tf.keras.optimizers.Adam):
    def __init__(self, initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9):
        self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate) 
        super().__init__(learning_rate=self.lr_scheduler)

    def lr_history_upto(self, iteration):
        iters = np.arange(0, iteration + 1)
        return self.lr_scheduler(iters)
        

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

    def call(self, inputs):  
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
    def __init__(self, inputs, outputs, activation=tf.tanh):
        super(Linear, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation

    def build(self, input_shape):  
        self.W = xavier_init([self.inputs, self.outputs]) 
        self.b = xavier_init([1, self.outputs])

    def call(self, inputs):  
        return self.activation(tf.add(tf.matmul(inputs, self.W), self.b))


def init_model(num_inputs, num_outputs, num_hidden_layers, num_neurons_per_layer, mean, variance):
    layer_sizes = [num_inputs] + [num_neurons_per_layer] * num_hidden_layers + [num_outputs] 
    layers = [Linear(layer_sizes[0], layer_sizes[1])] + \
             [ImprovedLinear(layer_sizes[i], layer_sizes[i + 1]) for i in range(1, len(layer_sizes) - 2)] + \
             [Linear(layer_sizes[-2], layer_sizes[-1], activation=tf.identity)]

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