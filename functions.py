import numpy as np
from matplotlib import pyplot as plt

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_images(images, cls_true, cls_pred=None, smooth=True):

    #assert len(images) == len(cls_true) == 100
    
    # Create figure with sub-plots.
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :])
            
        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
def plot_mnist(images, labels):
    # plot mnist images
    # images [num_images, 28, 28]
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_xlabel(np.matmul(labels[i], np.arange(10)))
    plt.show()


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

