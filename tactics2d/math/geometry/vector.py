import numpy as np


class Vector:
    """_summary_"""

    @staticmethod
    def angle(vec1, vec2) -> float:
        """Get the angle between two vectors.

        Returns:
            angle (float): The angle between two vectors in radians. The value is in the range [0, pi].
        """
        dot_product = np.dot(vec1, vec2)
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        cos_angle = dot_product / (vec1_norm * vec2_norm)
        angle = np.arccos(cos_angle)
        return angle
