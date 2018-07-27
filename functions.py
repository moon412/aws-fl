import numpy as np

def fedavg(clients_params):
    #clients_params is a list of list of arrays
    #Return: return a ndarray with dimension (n_layers, ...)
    #because the dimentionality of layers is different, the returned ndarray only has one dimention
    return np.array(clients_params).mean(axis=0)


#NUM_IMAGES_PER_CLIENT = 5000.0
NUM_CLIENTS = 10

def prob_dist(probs, probs_uniform=np.array([0.1]*10), norm=True):
    # Given a list of probabilities in probs:
    # Norm==False: compute the sum of square of probability difference
    # Norm==True: compute the 1-norm of probability difference
    # Difference compared to a uniform distribution 
    if norm:
        return np.linalg.norm(probs - probs_uniform, ord=1)
    else:
        return np.square(probs - probs_uniform).sum()

def create_uniform_probs(num_total, num_major):
    # 
    result = [(num_total - num_major)/(NUM_CLIENTS - 1.0)/num_total]*9
    result.append(num_major/num_total)
    return result

def target_function(x, y):
    return np.abs(np.abs(x[0]-0.1) + np.abs(x[1]-0.1) + np.abs(x[2]-0.1) + np.abs(x[3]-0.1) + np.abs(x[4]-0.1) +
                  np.abs(x[5]-0.1) + np.abs(x[6]-0.1) + np.abs(x[7]-0.1) + np.abs(x[8]-0.1) + np.abs(x[9]-0.1) - y)

cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[1] + x[2] + x[3] + x[4] +
                                         x[5] + x[6] + x[7] + x[8] + x[9] - 1.0})

