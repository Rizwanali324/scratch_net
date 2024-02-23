import numpy as np


def add_forward(a, b):
    return a.data + b.data


def add_backward(a, b, gradient):
    a.grad += gradient
    b.grad += gradient


def sub_forward(a, b):
    # TODO
    return a.data - b.data

    raise NotImplementedError()


def sub_backward(a, b, gradient):
    # TODO
    a.grad += gradient
    b.grad -= gradient
    raise NotImplementedError()


def mul_forward(a, b):
    # TODO
    return a.data * b.data

    raise NotImplementedError()


def mul_backward(a, b, gradient):
    # TODO
    a.grad += b.data * gradient
    b.grad += a.data * gradient
    raise NotImplementedError()


def div_forward(a, b):
    # TODO
    return a.data / b.data

    raise NotImplementedError()


def div_backward(a, b, gradient):
    # TODO
    a.grad += gradient / b.data
    b.grad -= (a.data * gradient) / (b.data ** 2)
    raise NotImplementedError()


def matmul_forward(a, b):
    # TODO
    return np.dot(a.data, b.data)

    raise NotImplementedError()


def matmul_backward(a, b, gradient):
    # TODO
    a.grad += np.dot(gradient, b.data.T)
    b.grad += np.dot(a.data.T, gradient)
    raise NotImplementedError()


def relu_forward(a):
    # TODO
    return np.maximum(0, a.data)

    raise NotImplementedError()


def relu_backward(a, gradient):
    # TODO
    a.grad += gradient * (a.data > 0)

    raise NotImplementedError()


def sigmoid_forward(a):
    # TODO
    return 1 / (1 + np.exp(-a.data))

    raise NotImplementedError()


def sigmoid_backward(a, gradient):
    # TODO
    sigmoid_a = 1 / (1 + np.exp(-a.data))
    a.grad += gradient * sigmoid_a * (1 - sigmoid_a)
    raise NotImplementedError()


def log_forward(a):
    # TODO
    return np.log(a.data)

    raise NotImplementedError()


def log_backward(a, gradient):
    # TODO
    a.grad += gradient / a.data

    raise NotImplementedError()


def nll_forward(scores, label):
    _scores = scores.data - np.max(scores.data)
    exp = np.exp(_scores)
    softmax_out = exp / np.expand_dims(np.sum(exp, axis=1), axis=1)

    mask = np.full(softmax_out.shape, False)
    for i, l in enumerate(label.data):
        mask[i, l] = True

    return -np.log(softmax_out[mask] + 1e-12)


def nll_backward(scores, label, gradient):
    _scores = scores.data - np.max(scores.data)
    exp = np.exp(_scores)
    softmax_out = exp / np.expand_dims(np.sum(exp, axis=1), axis=1)

    mask = np.full(softmax_out.shape, False)
    for i, l in enumerate(label.data):
        mask[i, l] = True

    grad = np.copy(softmax_out)
    grad[mask] -= 1

    scores.grad = grad * gradient
