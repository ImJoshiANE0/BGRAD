import math

class Value():
    """
    A scalar value node in a computational graph, used for building and 
    training neural networks through automatic differentiation.

    Each Value object stores:
    - `data`: the numerical scalar value.
    - `grad`: the gradient of the final output with respect to this value.
    - `_prev`: the set of child nodes that were used to compute this value.
    - `_op`: the operation that produced this node (for debugging/graph viz).
    - `_backward`: a function that applies the local gradient update during 
      backpropagation.

    The Value class supports operator overloading for mathematical operations 
    (addition, multiplication, exponentiation, division, negation, etc.), 
    enabling the construction of complex expressions using standard Python 
    syntax. Each operation creates a new Value node and records the computation 
    graph so gradients can later be propagated.

    This implementation is inspired by micrograd and is suitable for small-scale 
    neural network experiments or for educational purposes to understand 
    automatic differentiation and backpropagation mechanics.
    """
    
    def __init__(self, data, _children=(), _op=()):
        self.data = data # Numerical/scaler value
        self.grad = 0 # Gradiant of final expression w.r.t. self.data
        self._prev = set(_children) # List of Noded (Value scalers) that produced this node by doing some operation
        self._op = _op # Operation that produced this node
        self._backward = lambda: None # Method that propogates gradiant a single step backward
    
    # * operation is used in NN
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out

    # * operation is used in NN
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    # power is being defined to support multiple activation functions
    def __pow__(self, n):
        assert isinstance(n, (int, float)), "Valid power values are integers or floats!"
        out = Value(self.data ** n, (self,), f'**{n}')

        def _backward():
            self.grad += n * (self.data ** (n-1)) * out.grad

        out._backward = _backward
        return out

    # activation function tanh
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad = (1 - t**2) * out.grad

        out._backward = _backward
        return out

    # activation function relu
    def relu(self):
        # gradient of relu(x) wrt x is 1 if x > 0 else 0
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        
        out._backward = _backward
        return out

    # To perform backward propagation from this node to all previous nodes.
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"