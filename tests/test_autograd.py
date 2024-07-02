import pytest
import numpy as np
from pandagrad.ptensor import PTensor, Add, Mul

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

def test_combined_operations():
    a = PTensor([2.0], requires_grad=True)
    b = PTensor([3.0], requires_grad=True)
    c = a + b  
    d = a * b  
    e = c + d  
    e.backward()
    
    assert np.allclose(e.data, [11.0]), f"Expected e.data to be [11.0], but got {e.data}"
    assert np.allclose(a.grad, [4.0]), f"Expected a.grad to be [4.0], but got {a.grad}"
    assert np.allclose(b.grad, [3.0]), f"Expected b.grad to be [5.0], but got {b.grad}"

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
