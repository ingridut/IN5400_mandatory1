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

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1.1
    params = {}
    layer_count = 1
    lmin1 = conf['layer_dimensions'][0]
    
    
    for l in conf['layer_dimensions'][1:]:
        params['W_'+str(layer_count)] = np.random.normal(loc=0, scale=2/lmin1, size = (lmin1, l))
        params['b_'+str(layer_count)] = np.zeros(shape=(l, 1))
        lmin1 = l
        layer_count += 1
        
    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 a)
    if activation_function == 'relu':
        Z[Z<=0]=0
        return Z
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # Subtract max(Z) for stability 
    exp_Z = np.exp(Z - np.amax(Z, axis=0, keepdims=True))
    
    return exp_Z/np.sum(exp_Z, axis=0)


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 1.2 c)
    Y_proposed = None
    features = {}
    
    A = X_batch
    features['A_0'] = A
    

    for l in range(len(conf['layer_dimensions'])-2):
        Z = params['W_'+str(l+1)].T.dot(A) + np.outer(params['b_'+str(l+1)], np.ones(np.shape(A)[1]))
        features['Z_'+str(l+1)] = np.copy(Z)

        A = activation(Z, conf['activation_function'])
        features['A_'+str(l+1)] = A

    L = len(conf['layer_dimensions'])-1
    Z = params['W_'+str(L)].T.dot(A) + np.outer(params['b_'+str(L)], np.ones(np.shape(A)[1]))
    features['Z_'+str(L)] = np.copy(Z)        
    Y_proposed = softmax(Z)
    
    if not(is_training):
        features = None

    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 1.3
    # Predict Y by setting largest propbability to 1    
    Y_predicted = (Y_proposed == Y_proposed.max(axis=0, keepdims=1)).astype(int)
    num_correct = np.sum(np.all(Y_predicted == Y_reference, axis=0))
    
    # Calculate cost 
    log_prop = np.log(Y_proposed)
    cost = - (1/np.shape(Y_proposed)[1]) * np.sum(np.multiply(Y_reference, log_prop))

    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.4 a)
    Z_C = Z.copy()
    if activation_function == 'relu':
        Z_C[Z_C >= 0] = 1
        Z_C[Z_C < 0] = 0
        return Z_C
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 1.4 b)
    loss = Y_proposed - Y_reference
    
    num_layers = len(conf['layer_dimensions'])
    m = np.shape(Y_proposed)[1]
    grad_params = {}
    
    for l in reversed(range(1, num_layers)):
        grad_params['grad_W_'+str(l)] = (1/m) * features['A_'+str(l-1)].dot(loss.T)
        grad_params['grad_b_'+str(l)] = (1/m) * loss.dot(np.ones(shape=(np.shape(loss)[1], 1)))
        
        if l > 1:
            # update loss
            dw_a = activation_derivative(features['Z_'+str(l-1)], activation_function='relu')
            prev_loss = loss
            loss = np.multiply((params['W_' + str(l)].dot(prev_loss)), dw_a)
        else:
            break
        
    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    # TODO: Task 1.5
    updated_params = {}
    num_layers = int(len(params.keys())/2)
    for l in range(1, num_layers+1):
        updated_params['W_' + str(l)] = params['W_' + str(l)] - conf['learning_rate']*grad_params['grad_W_' + str(l)]
        updated_params['b_' + str(l)] = params['b_' + str(l)] - conf['learning_rate']*grad_params['grad_b_' + str(l)]
    return updated_params
