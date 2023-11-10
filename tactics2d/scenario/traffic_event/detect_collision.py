from shapely import Geometry


class CollisionDetector:
    @staticmethod
    def detect_static_collision(trace: Geometry, obstacles: Geometry) -> bool:
        """Detect if the agent collides with the static obstacles."""
        
        return False

    @staticmethod
    def detect_dynamic_collision(*args):
        return
