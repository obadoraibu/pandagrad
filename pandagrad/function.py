import numpy as np
from .ptensor import PTensor, Function

class ReLU(Function):
    def forward(self, a: PTensor) -> PTensor:
        self.a = a
        return PTensor(np.maximum(0, a.data), a.requires_grad)
    
    def backward(self, grad: np.ndarray):
        if self.a.requires_grad:
            print(self.a.data)
            print((self.a.data > 0).astype(grad.dtype))
            self.a.backward(grad * (self.a.data > 0).astype(grad.dtype))