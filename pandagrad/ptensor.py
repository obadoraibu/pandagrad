import numpy as np

class PTensor:
    def __init__(self, data, requires_grad: bool = False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._grad_fn = None

    def set_fn(self, function):
        self._grad_fn = function

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float32)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self._grad_fn is not None:
            self._grad_fn.backward(grad)

    def __add__(self, other):
        return Add.apply(self, other)

    def __mul__(self, other):
        return Mul.apply(self, other)
    
    def __sub__(self, other):
        return Sub.apply(self, other)
    
    def __truediv__(self, other):
        return Div.apply(self, other)
    
    def __matmul__(self, other):
        return MatMul.apply(self, other)

class Function:
    @classmethod
    def apply(cls, *inputs):
        obj = cls()
        obj.inputs = inputs
        obj.output = obj.forward(*inputs)
        obj.output.set_fn(obj)
        return obj.output

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError
    
    
class Add(Function):
    def forward(self, a: PTensor, b: PTensor) -> PTensor:
        self.a = a
        self.b = b
        return PTensor(a.data + b.data, requires_grad=(a.requires_grad or b.requires_grad))

    def backward(self, grad: np.ndarray):
        if self.a.requires_grad:
            self.a.backward(grad)
        if self.b.requires_grad:
            self.b.backward(grad)

class Mul(Function):
    def forward(self, a: PTensor, b: PTensor) -> PTensor:
        self.a = a
        self.b = b
        return PTensor(a.data * b.data, requires_grad=(a.requires_grad or b.requires_grad))

    def backward(self, grad: np.ndarray):
        if self.a.requires_grad:
            self.a.backward(grad * self.b.data)
        if self.b.requires_grad:
            self.b.backward(grad * self.a.data)

class Sub(Function):
    def forward(self, a: PTensor, b: PTensor) -> PTensor:
        self.a = a
        self.b = b
        return PTensor(a.data - b.data, requires_grad=(a.requires_grad or b.requires_grad))

    def backward(self, grad: np.ndarray):
        if self.a.requires_grad:
            self.a.backward(grad)
        if self.b.requires_grad:
            self.b.backward(-grad)
            
class Div(Function):
    def forward(self, a: PTensor, b: PTensor) -> PTensor:
        self.a = a
        self.b = b
        return PTensor(a.data / b.data, requires_grad=(a.requires_grad or b.requires_grad))

    def backward(self, grad: np.ndarray):
        if self.a.requires_grad:
            self.a.backward(grad / self.b.data)
        if self.b.requires_grad:
            self.b.backward(-grad * self.a.data / (self.b.data ** 2))
            
class MatMul(Function):
    def forward(self, a: PTensor, b: PTensor) -> PTensor:
        self.a = a
        self.b = b
        return PTensor(a.data @ b.data, requires_grad=(a.requires_grad or b.requires_grad))
    
    def backward(self, grad: np.ndarray):
        if self.a.requires_grad:
            self.a.backward(grad @ self.b.data.T)
        if self.b.requires_grad:
            self.b.backward(self.a.data.T @ grad)   
            
