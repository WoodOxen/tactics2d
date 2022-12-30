import numpy as np

def get_angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Get the angle value between two vectors in radians [0, pi].
    """
    len1 = np.linalg.norm(vec1)
    len2 = np.linalg.norm(vec2)
    theta = np.arccos(np.dot(vec1, vec2) / (len1 * len2))
    return theta


def get_angle_sign(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Get the sign of the angle between two vectors
    """
    return np.sign(np.cross(vec1, vec2))