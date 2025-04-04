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
    def __init__(self, bvp, optimizer, num_hidden_layers=4, num_neurons_per_layer=50, activation=tf.tanh):
        """
        Constructor for a boundary value problem solver.

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

        Examples
        -----------
        >>> from PDESolver import *
        >>> optimizer = Optimizer(initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9)
        >>> bvp = Laplace()
        >>> solver = Solver(bvp, optimizer, num_hidden_layers=4, num_neurons_per_layer=50)
        """

        num_inputs = len(bvp.get_specification()['variables'])
        num_outputs = len(bvp.get_specification()['components'])
        inner_constraint = [condition for condition in bvp.get_conditions() if condition.name == 'inner'][0]
        mean, variance = inner_constraint.get_normalization_constants()

        self.model = init_model(num_inputs, num_outputs, num_hidden_layers, num_neurons_per_layer, mean, variance, activation)
        self.bvp = bvp
        self.optimizer = optimizer

        self.loss_history = []
        self.weight_history = [{cond.name: 1. for cond in bvp.get_conditions()}]
        self.weights = tf.Variable(np.ones(len(bvp.get_conditions())), dtype=tf.float32)
        self.step = tf.Variable(0)

    def compute_differentials(self, samplePoints):
        """
        Calculates the differentials of the model at the given points.

        Parameters
        -----------
        samplePoints: tensor
            Tensor of points to calculate the differentials at

        Returns
        -----------
        dict: Dictionary containing the differentials of the model needed for the bvp at the given points

        Examples
        -----------
        >>> from PDESolver import *
        >>> solver = Solver(Laplace(), Optimizer())
        >>> samples = tf.constant([[0, 0], [0, 0.5], [0, 1]])  
        >>> solver.compute_differentials(samples) 
        {'x': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
        array([[0.],
            [0.],
            [0.]], dtype=float32)>, 'y': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
        array([[0. ],
            [0.5],
            [1. ]], dtype=float32)>, 'u': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
        array([[-1.4298753 ],
            [-1.0829226 ],
            [-0.86016774]], dtype=float32)>, 'u_x': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
        array([[0.7101533 ],
            [0.7047943 ],
            [0.45764494]], dtype=float32)>, 'u_xx': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
        array([[-1.6525286],
            [-1.6283079],
            [-1.2708694]], dtype=float32)>, 'u_y': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
        array([[0.71874154],
            [0.6189531 ],
            [0.2856549 ]], dtype=float32)>, 'u_yy': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
        array([[-0.01344297],
            [-0.5869958 ],
            [-0.47277603]], dtype=float32)>}
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

        Examples
        -----------
        >>> from PDESolver import *
        >>> solver = Solver(Laplace(minibatch_size=3), Optimizer()) 
        >>> solver.compute_residuals()
        {'zero_boundary': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
        array([[1.7787822],
            [2.6922252],
            [1.3558891]], dtype=float32)>, 'f_boundary': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
        array([[2.4186134],
            [2.4582624],
            [2.4142814]], dtype=float32)>, 'inner': <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
        array([[ 0.53346014],
            [-4.9377484 ],
            [ 0.21452093],
            [-1.6314204 ]], dtype=float32)>}
        """

        conditions = self.bvp.get_conditions()
        fun_name = self.bvp.get_specification()["components"][0]

        residuals = {}
        for condition in conditions:
            samples = condition.sample_points()
            Du = self.compute_differentials(samples)
            residuals[condition.name] = condition.residue_fn(Du) + 0*Du[fun_name] # Why this helps? IDK??!?!?! wtf

        return residuals
  
    def compute_losses(self):
        """
        Computes the losses for the neural network.
        
        Returns
        -----------
        dict: Dictionary of losses of form {condition_name: loss}

        Examples
        -----------
        >>> from PDESolver import *
        >>> solver = Solver(Laplace(), Optimizer()) 
        >>> solver.compute_losses() 
        {'zero_boundary': <tf.Tensor: shape=(), dtype=float32, numpy=0.29857153>, 'f_boundary': <tf.Tensor: shape=(), dtype=float32, numpy=0.07588613>, 'inner': <tf.Tensor: shape=(), dtype=float32, numpy=6.3834033>, 'L2': <tf.Tensor: shape=(), dtype=float32, numpy=6.757861>, 'total': <tf.Tensor: shape=(), dtype=float32, numpy=13.515722>}
        """

        criterion = tf.keras.losses.MeanSquaredError()
        residuals = self.compute_residuals()

        losses = {name: 0.0 for name in residuals.keys()}
        l2loss = 0.0
        for i, (name, residual) in enumerate(residuals.items()):
            losses[name] += self.weights[i] * criterion(residual, 0.0)
            l2loss += criterion(residual, 0.0)

        losses['total'] = sum(losses.values())
        losses['L2'] = l2loss

        return losses
 
    def compute_gradients(self):
        """
        Gets the gradients of the neural network.

        Returns
        -----------
        tensor: L2 loss
        dict: Dictionary of gradients of form {condition_name: gradient}

        Examples
        -----------
        >>> from PDESolver import *
        >>> solver = Solver(Laplace(), Optimizer())
        >>> solver.compute_gradients()
        (<tf.Tensor: shape=(), dtype=float32, numpy=4.756295>, LARGE DICTIONARY)
        """

        with tf.GradientTape(persistent=True) as tape:
            losses = self.compute_losses()

            kwargs = {'sources': self.model.trainable_variables, 'unconnected_gradients': tf.UnconnectedGradients.ZERO}

            grads = {name: tape.gradient(loss, **kwargs) for name, loss in losses.items() if name != 'L2'}

        return losses['L2'], grads


    def adjust_weights(self, gradients):
        """
        Adjusts the weights of the PDE and data losses.

        Parameters
        -----------
        gradients: dict
            Dictionary of gradients as returned by compute_gradients()

        Returns
        -----------
        dict: Dictionary of adjusted condition weights of form {condition_name: weight}

        Examples
        -----------
        >>> from PDESolver import *
        >>> solver = Solver(Laplace(), Optimizer()) 
        >>> l2, gradients = solver.compute_gradients()
        >>> solver.adjust_weights(gradients)
        {'zero_boundary': <tf.Tensor: shape=(), dtype=float32, numpy=38.457893>, 'f_boundary': <tf.Tensor: shape=(), dtype=float32, numpy=22.964037>, 'inner': <tf.Tensor: shape=(), dtype=float32, numpy=1.0>}
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
                new_weight = 0.9 * self.weights[i] + 0.1 * new_weight
            else:
                new_weight = self.weights[i]

            new_weights[name] = new_weight
            minimal = tf.math.minimum(minimal, new_weights[name])

        new_weights = {name: new_weights[name] / minimal for name in new_weights.keys()}
        self.weights.assign([new_weight for new_weight in new_weights.values()])
                
        return new_weights
    
    @tf.function
    def train_step(self):
        """
        Performs a single training step.

        Returns
        -----------
        tensor: L2 loss
        dict: Dictionary of gradients of form {condition_name: gradient}
        dict: Dictionary of adjusted condition weights of form {condition_name: weight}

        Examples
        -----------
        >>> from PDESolver import *
        >>> solver = Solver(Laplace(), Optimizer())
        >>> l2loss, gradients, new_weights = solver.train_step()
        >>> l2loss
        <tf.Tensor: shape=(), dtype=float32, numpy=4.371607>
        >>> l2loss, gradients, new_weights = solver.train_step()
        >>> l2loss
        <tf.Tensor: shape=(), dtype=float32, numpy=3.0963678>
        >>> l2loss, gradients, new_weights = solver.train_step()
        >>> l2loss
        <tf.Tensor: shape=(), dtype=float32, numpy=3.029224>
        """
        
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

        Examples
        -----------
        >>> from PDESolver import *
        >>> solver = Solver(Laplace(), Optimizer())
        >>> solver.train(iterations=10000, debug_frequency=2500)
        øL²-loss = 6.044e+01 (best: 2.305e+01, 000064it ago) lr = 0.00090:   3%|█                                | 1271/40000 [00:33<07:53, 81.76it/s]
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
            
            avg_loss = 10**np.mean(np.log10(self.loss_history[-100:]))
            pbar.desc = f'øL²-loss = {avg_loss:.3e} (best: {best_loss:.3e}, {iterations_since_last_improvement:0{k_max}d}it ago) lr = {self.optimizer.lr.numpy():.5f}'

            if debug_frequency > 0 and (i % debug_frequency == 0 or i == iterations - 1):
                self.show_debugplot(gradients)

    def evaluate(self, values):
        """
        Evaluates the neural network at a given set of values.

        Parameters
        -----------
        values: object
            object with variable names as keys and lists of values to evaluate at as values 

        Returns
        -----------
        tensor: Tensor of shape (len(values), 1) containing the evaluated values

        Examples
        -----------
        >>> optim = Optimizer(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.98)
        >>> solver = Solver(BlackScholes(), optim, num_hidden_layers=4, num_neurons_per_layer=50)
        >>> solver.train(iterations=N, debug_frequency=N)
        >>> solver.evaluate({'t': 0, 'S': 15})
        >>> XXX
        """

        bvp_variables = self.bvp.get_specification()["variables"]

        # convert all inputs to lists
        for name in values.keys():
            if type(values[name]) != list:
                values[name] = [values[name]]

        values_lengths = [len(values) for values in values.values()]

        assert len(set(values_lengths)) == 1, "All values must have the same length"

        assert set(values.keys()) == set(bvp_variables), "Provided variables must match with the ones specified in the BVP"

        model_input = tf.transpose(tf.convert_to_tensor([[values[name] for name in bvp_variables]], dtype=tf.float32))

        return self.model(model_input).numpy()

    def show_debugplot(self, gradients):
        """
        Shows a debug plot of the neural network.

        Parameters
        -----------
        gradients: tuple
            Tuple of gradients of the PDE and data losses obtained from compute_gradients

        Examples
        -----------
        >>> from PDESolver import *                        
        >>> solver = Solver(Laplace(), Optimizer())
        >>> l2loss, gradients = solver.compute_gradients()
        >>> solver.loss_history += [l2loss.numpy()]
        >>> solver.show_debugplot(gradients)    
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
    """"""
    def __init__(self, initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9):
        """
        Optimizer for the neural network used in a solver object.
        Acts as a wrapper for the Adam optimizer with an exponential learning rate decay.

        Parameters
        -----------
        initial_learning_rate: float
            Initial learning rate of the optimizer.
            Defaults to 1e-3.
        decay_steps: int
            Number of iterations after which the learning rate has decayed by the specified factor.
            Defaults to 1000.
        decay_rate: float
            Factor by which the learning rate is decayed.
            Defaults to 0.9.

        Examples
        -----------
        >>> from PDESolver import *
        >>> optimizer = Optimizer(initial_learning_rate=1, decay_steps=500, decay_rate=0.5)
        >>> solver = Solver(Laplace(), optimizer)
        """

        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate)
        super().__init__(learning_rate=lr_scheduler)
        self.lr_scheduler = lr_scheduler

    @property
    def lr(self):
        """
        Returns the current learning rate of the optimizer.

        Returns
        -----------
        float: Current learning rate of the optimizer.
        """

        return self.lr_scheduler(self.iterations)

    def lr_history_upto(self, iteration):
        """
        Returns the learning rate history up to the specified iteration.

        Parameters
        -----------
        iteration: int
            Iteration up to which the learning rate history is returned.

        Returns
        -----------
        numpy.ndarray: Learning rate history up to the specified iteration.

        Examples
        -----------
        >>> optimizer = Optimizer()
        >>> optimizer.lr_history_upto(10)
        array([0.001     , 0.00099989, 0.00099979, 0.00099968, 0.00099958,
               0.00099947, 0.00099937, 0.00099926, 0.00099916, 0.00099905,
               0.00099895], dtype=float32)>
        """

        iters = np.arange(0, iteration + 1)
        return self.lr_scheduler(iters)
        

###################################################################################
###################################################################################
###################################################################################

                
def xavier_init(self,name,size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
    return self.add_weight(name=name, shape=size, initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=xavier_stddev))


class Encoder(tf.keras.layers.Layer):
    def __init__(self, inputs, outputs, activation=tf.tanh):
        super(Encoder, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation

    def build(self, input_shape):  
        self.W = xavier_init(self, "W", [self.inputs, self.outputs])
        self.b = xavier_init(self, "b", [1, self.outputs])
        
    def call(self, inputs):  
        return self.activation(tf.add(tf.matmul(inputs, self.W), self.b))
    

class ImprovedLinear(tf.keras.layers.Layer):
    def __init__(self, inputs, outputs, activation=tf.tanh):
        super(ImprovedLinear, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation

    def build(self, input_shape):  
        self.W = xavier_init(self, "W", [self.inputs, self.outputs])
        self.b = xavier_init(self, "b", [1, self.outputs])

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
        self.W = xavier_init(self, "W", [self.inputs, self.outputs])
        self.b = xavier_init(self, "b", [1, self.outputs])

    def call(self, inputs):  
        return self.activation(tf.add(tf.matmul(inputs, self.W), self.b))


def init_model(num_inputs, num_outputs, num_hidden_layers, num_neurons_per_layer, mean, variance, activation=tf.tanh):
    layer_sizes = [num_inputs] + [num_neurons_per_layer] * num_hidden_layers + [num_outputs] 
    layers = [Linear(layer_sizes[0], layer_sizes[1], activation=activation)] + \
             [ImprovedLinear(layer_sizes[i], layer_sizes[i + 1], activation=activation) for i in range(1, len(layer_sizes) - 2)] + \
             [Linear(layer_sizes[-2], layer_sizes[-1], activation=tf.identity)]

    encoder_1 = Encoder(num_inputs, num_neurons_per_layer)
    encoder_2 = Encoder(num_inputs, num_neurons_per_layer)

    inputs = tf.keras.Input((num_inputs,))
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

    return model


if __name__ == '__main__':
    from Sampling import Cuboid, Equidistant

    tf.random.set_seed(1)
    txs = Cuboid([0, 0], [1, 1]).pick(9, Equidistant())

    model = init_model(2, 1, 4, 50, mean=-1, variance=1)

    model.summary()
    print(model(txs)) 