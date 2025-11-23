import random
from core.value import Value

# Module that enforces certain properties in Neural Network classes
class BaseNeuralModule:
    """
    Base class for all neural network components.

    This class defines the minimal interface required for trainable modules:
    - `parameters()`: returns all Value objects that represent learnable
      parameters of the module.
    - `zero_grad()`: resets gradients for every parameter returned by
      `parameters()`.

    Subclasses are expected to override `parameters()` and implement their own
    forward logic through `__call__`.
    """

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(BaseNeuralModule):
    """
    A single fully connected neuron with a tanh activation.

    The neuron holds:
    - a weight Value for each input dimension
    - a bias Value

    When called, it computes a weighted sum of the inputs, adds the bias,
    and applies the tanh activation. All computations are tracked using the
    Value class for automatic differentiation.
    """

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    
    def __call__(self, inputs):
        # print(list(zip(self.w, inputs)))
        res = sum((wi*xi for wi, xi in zip(self.w, inputs)), self.b)
        act = res.tanh()
        return act
    
    def parameters(self):
        return self.w + [self.b]

class Layer(BaseNeuralModule):
    """
    A fully connected neural network layer composed of multiple Neuron objects.

    Each layer contains `nouts` neurons, each receiving `nin` inputs.
    Calling the layer forwards the input through every neuron and returns
    either a list of outputs or a single Value if the layer has one neuron.
    """

    def __init__(self, nin, nouts):
        self.neurons = [Neuron(nin) for _ in range(nouts)]

    def __call__(self, inputs):
        out = [n(inputs) for n in self.neurons]
        return out[0] if len(out)==1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(BaseNeuralModule):
    """
    A multilayer perceptron built from a sequence of fully connected layers.

    The network is defined by an input size and a list of layer sizes.
    Each layer uses tanh activation internally (in its Neuron units).
    Calling the MLP forwards the input through each layer in order and
    returns the final output.

    All learnable parameters of all layers are aggregated through `parameters()`
    for use in training.
    """

    def __init__(self, nin, nlayers):
        sz = [nin] + nlayers
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nlayers))]
    
    def __call__(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]