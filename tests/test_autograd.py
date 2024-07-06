import pytest
import numpy as np
from pandagrad.ptensor import PTensor, Add, Mul, Sub, Div, MatMul
from pandagrad.function import ReLU

def test_addition():
    a = PTensor([2.0], requires_grad=True)
    b = PTensor([3.0], requires_grad=True)
    c = a + b 
    c.backward()
    
    assert np.allclose(c.data, [5.0]), f"Expected c.data to be [5.0], but got {c.data}"
    assert np.allclose(a.grad, [1.0]), f"Expected a.grad to be [1.0], but got {a.grad}"
    assert np.allclose(b.grad, [1.0]), f"Expected b.grad to be [1.0], but got {b.grad}"

def test_multiplication():
    a = PTensor([2.0], requires_grad=True)
    b = PTensor([3.0], requires_grad=True)
    d = a * b  
    d.backward()
    
    assert np.allclose(d.data, [6.0]), f"Expected d.data to be [6.0], but got {d.data}"
    assert np.allclose(a.grad, [3.0]), f"Expected a.grad to be [3.0], but got {a.grad}"
    assert np.allclose(b.grad, [2.0]), f"Expected b.grad to be [2.0], but got {b.grad}"

def test_subtraction():
    a = PTensor([5.0], requires_grad=True)
    b = PTensor([3.0], requires_grad=True)
    c = a - b 
    c.backward()
    
    assert np.allclose(c.data, [2.0]), f"Expected c.data to be [2.0], but got {c.data}"
    assert np.allclose(a.grad, [1.0]), f"Expected a.grad to be [1.0], but got {a.grad}"
    assert np.allclose(b.grad, [-1.0]), f"Expected b.grad to be [-1.0], but got {b.grad}"

def test_division():
    a = PTensor([6.0], requires_grad=True)
    b = PTensor([3.0], requires_grad=True)
    c = a / b 
    c.backward()
    
    assert np.allclose(c.data, [2.0]), f"Expected c.data to be [2.0], but got {c.data}"
    assert np.allclose(a.grad, [1.0 / 3.0]), f"Expected a.grad to be [1.0 / 3.0], but got {a.grad}"
    assert np.allclose(b.grad, [-2.0 / 3.0]), f"Expected b.grad to be [-2.0 / 3.0], but got {b.grad}"

def test_matrix_multiplication():
    a = PTensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = PTensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)
    c = a @ b 
    c.backward(np.ones_like(c.data))
    
    assert np.allclose(c.data, [[4.0, 4.0], [10.0, 8.0]]), f"Expected c.data to be [[4.0, 4.0], [10.0, 8.0]], but got {c.data}"
    assert np.allclose(a.grad, [[2.0, 3.0], [2.0, 3.0]]), f"Expected a.grad to be [[2.0, 3.0], [2.0, 3.0]], but got {a.grad}"
    assert np.allclose(b.grad, [[4.0, 4.0], [6.0, 6.0]]), f"Expected b.grad to be [[4.0, 4.0], [6.0, 6.0]], but got {b.grad}"

def test_combined_operations():
    a = PTensor([2.0], requires_grad=True)
    b = PTensor([3.0], requires_grad=True)
    c = a + b  
    d = a * b  
    e = c + d  
    e.backward()
    
    assert np.allclose(e.data, [11.0]), f"Expected e.data to be [11.0], but got {e.data}"
    assert np.allclose(a.grad, [4.0]), f"Expected a.grad to be [4.0], but got {a.grad}"
    assert np.allclose(b.grad, [3.0]), f"Expected b.grad to be [3.0], but got {b.grad}"

def test_relu_forward():
    a = PTensor([-1.0, 0.0, 1.0], requires_grad=True)
    relu = ReLU()
    b = relu.forward(a)
    
    assert np.allclose(b.data, [0.0, 0.0, 1.0]), f"Expected b.data to be [0.0, 0.0, 1.0], but got {b.data}"

def test_relu_backward():
    a = PTensor([-1.0, 0.0, 1.0], requires_grad=True)
    relu = ReLU()
    b = relu.apply(a)
    grad = np.array([1.0, 1.0, 1.0])
    b.backward(grad)
    
    assert np.allclose(a.grad, [0.0, 0.0, 1.0]), f"Expected a.grad to be [0.0, 0.0, 1.0], but got {a.grad}"


if __name__ == "__main__":
    pytest.main([__file__])
