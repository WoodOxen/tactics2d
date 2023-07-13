import numpy as np
np.random.seed(42)

def truncate_gaussian(mean, std, min_, max_, size=None):
    """Return a random number from a truncated gaussian distribution.

    The distribution is truncated to the interval [min, max]. The distribution is defined
        by the mean and standard deviation of the gaussian distribution.

    Args:
        mean (float or array_like of floats): The mean of the gaussian distribution.
        std (float or array_like of floats): The standard deviation of the gaussian distribution.
        min_ (float or array_like of floats): The minimum value of the truncated gaussian distribution.
        max_ (float or array_like of floats): The maximum value of the truncated gaussian distribution.
        size (int or tuple of ints): The number of random numbers to generate.
    """
    random_numbers = np.random.normal(mean, std, size)
    return np.clip(random_numbers, min_, max_)
