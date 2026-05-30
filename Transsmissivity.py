import numpy as np


def transmissivity(return_number, numbers_of_return, nearest_neighbors):

    """Calculate the transmissivity for each point based on its nearest neighbors and return information.
    Parameters:
    return_number (np.ndarray): An array of shape (n_samples,) containing the return number for each point.
    numbers_of_return (np.ndarray): An array of shape (n_samples,) containing the total number of returns for each point.
    nearest_neighbors (np.ndarray): An array of shape (n_samples, n_neighbors) containing the indices of the nearest neighbors for each point.
    Returns:
    np.ndarray: An array of transmissivity values for each point.
    """

    first_intermediate_returns = return_number != numbers_of_return
    t = np.zeros(nearest_neighbors.shape[0])
    for i in range(nearest_neighbors.shape[0]):
        nn = nearest_neighbors[i]
        t[i] = first_intermediate_returns[nn].sum()/len(nn)
    return t

