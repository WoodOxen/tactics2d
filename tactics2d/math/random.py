import numpy as np


def truncnorm(mean, std, size, min, max):
    """Return a random number from a truncated normal distribution.

    The distribution is truncated to the interval [min, max]. The distribution is defined 
        by the mean and standard deviation of the normal distribution.

    Args:
        mean (float or array_like of floats): The mean of the normal distribution.
        std (float or array_like of floats): The standard deviation of the normal distribution.
        size (int or tuple of ints): The number of random numbers to generate.
        min (float or array_like of floats): The minimum value of the truncated normal distribution.
        max (float or array_like of floats): The maximum value of the truncated normal distribution.
    """
    random_number = np.random.normal(mean, std, size)
    return np.clip(random_number, min, max)