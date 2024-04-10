from transformer import layers
import torch
from torch import nn

def test_Linear():
    # layers
    my = layers.Linear(2,3)
    py = nn.Linear(2,3)

    py.weight, py.bias = my.weight, my.bias

    # inputs
    inputs = [
        torch.rand(2),
        torch.rand(4,2), 
        torch.rand(4,5,2)
    ]

    # tests
    for x in inputs:
        assert torch.allclose(my(x), py(x))

def test_Embedding():
    # layers
    my = layers.Embedding(3,4)
    py = nn.Embedding(3,4)

    py.weight = my.weight

    # inputs
    inputs = [
        torch.randint(0, 3, (3,)),
        torch.randint(0, 3, (2,3)),
        torch.randint(0, 3, (2,3,4)),
    ]

    # tests
    for x in inputs:
        assert torch.allclose(my(x), py(x))

def test_LayerNorm():
    # layers
    my = layers.LayerNorm(3)
    py = nn.LayerNorm(3)

    # inputs
    inputs = [
        torch.rand(3),
        torch.rand(2,3),
        torch.rand(4,2,3)
    ]

    # tests
    for x in inputs:
        assert torch.allclose(my(x), py(x), rtol=1e-04, atol=1e-06)

def test_Softmax():
    # layers
    my = layers.Softmax(-1)
    py = nn.Softmax(-1)

    # inputs
    inputs = [
        torch.rand(3),
        torch.rand(2,3),
        torch.rand(4,2,3)
    ]

    # tests
    for x in inputs:
        assert torch.allclose(my(x), py(x))

def test_GELU():
    # layers
    my = layers.GELU()
    py = nn.GELU(approximate='tanh')

    # inputs
    inputs = [
        torch.rand(3),
        torch.rand(2,3),
        torch.rand(4,2,3)
    ]

    # tests
    for x in inputs:
        assert torch.allclose(my(x), py(x))

def test_Sequential():
    # layers
    my_0 = layers.Linear(3,4)
    my_1 = layers.GELU()
    my_2 = layers.Linear(4,2)

    py_0 = nn.Linear(3,4)
    py_1 = nn.GELU(approximate='tanh')
    py_2 = nn.Linear(4,2)

    py_0.weight, py_0.bias = my_0.weight, my_0.bias
    py_2.weight, py_2.bias = my_2.weight, my_2.bias

    my = layers.Sequential(my_0, my_1, my_2)
    py = nn.Sequential(py_0, py_1, py_2)

    # inputs
    inputs = [
        torch.rand(3),
        torch.rand(2,3),
        torch.rand(4,2,3)
    ]

    # tests
    for x in inputs:
        assert torch.allclose(my(x), py(x))

def test_CrossEntropyLoss():
    # layers
    my = layers.CrossEntropyLoss()
    py = nn.CrossEntropyLoss()

    # inputs
    inputs = [
        torch.rand(5),
        torch.rand(3,5),
        torch.rand(3,5,7)
    ]

    targets = [
        torch.randint(0,5,()),
        torch.randint(0,5,(3,)),
        torch.randint(0,5,(3,7))
    ]

    # ignore indices
    targets[1][1] = -100
    targets[2][1,4] = -100

    # tests
    for input, target in zip(inputs, targets):
        assert torch.allclose(my(input, target), py(input, target))