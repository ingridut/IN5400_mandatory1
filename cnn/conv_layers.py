#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 1 in                                             #
# IN5400 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.02.12                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Implementation of convolution forward and backward pass"""

import numpy as np

def conv_layer_forward(input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_alyer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """
    # TODO: Task 2.1
    

    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    (num_filters, channels_w, height_w, width_w) = weight.shape
    
    pad_w = ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size))
    input_padded = np.pad(input_layer, pad_width = pad_w, mode='constant', constant_values=0)
    
    height_o = int(1 + (height_x + 2*pad_size - height_w)/stride)
    width_o = int(1 + (width_x + 2*pad_size - width_w)/stride)
    
    output_layer = np.zeros(shape=(batch_size, num_filters, height_o, width_o)) # Should have shape (batch_size, num_filters, height_y, width_y)

    
    hw_k = height_w // 2
    ww_k = width_w // 2

    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")
    
    for b in range(batch_size):
        for n in range(num_filters):
            for x in range(0, height_x, stride):
                for y in range(0, width_x, stride):
                    for c in range(channels_x):
                        output_layer[b, n, int(x/stride), int(y/stride)] += np.sum(input_padded[b, c, x:x+height_w, y:y+width_w]*weight[n, c, :, :])
            # Add bias
            output_layer[b, n, :, :] += bias[n]
            
    return output_layer


def conv_layer_backward(output_layer_gradient, input_layer, weight, bias, pad_size=1):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        input_layer_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    # TODO: Task 2.2
    input_layer_gradient, weight_gradient, bias_gradient = None, None, None

    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    num_filters, channels_w, height_w, width_w = weight.shape

    bias_gradient = np.zeros(bias.shape)
    input_layer_gradient = np.zeros(input_layer.shape)

    weight_gradient = np.zeros(weight.shape)
    
    pad_w = ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size))
    input_padded = np.pad(input_layer, pad_width = pad_w, mode='constant', constant_values=0)
    
    K = height_w // 2
    
    assert num_filters == channels_y, (
        "The number of filters must be the same as the number of output layer channels")
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")
    
    for b in range(batch_size):
        for n in range(num_filters):
            for c in range(channels_x):
                for r in range(height_w):
                    for s in range(width_w):
                        weight_gradient[n, c, r, s] += np.sum(np.multiply(output_layer_gradient[b, n, :, :], input_padded[b, c, r:width_x+r, s:width_x+s]))
                        
            for k in range(channels_w):
                # input layer gradient equal to convolution of output_layer_gradient convolution with weights turned 180 degrees
                input_layer_gradient[b, k, :, :] += conv_layer_forward(output_layer_gradient[b, n, :, :].reshape(1, 1, height_y, width_y), np.rot90(weight[n, k, :, :], 2, (0, 1)).reshape(1, 1, height_w, width_w), bias=np.zeros(1)).reshape(height_x, width_x)
            bias_gradient[n] += np.sum(output_layer_gradient[b, n, :, :])
            
         
    return input_layer_gradient, weight_gradient, bias_gradient


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
