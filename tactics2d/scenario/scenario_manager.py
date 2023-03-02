from .traffic_event import TrafficEvent


class ScenarioManager(object):
    """This class detects traffic status in the given traffic scenario.

    Attributes:
        map_ (Map): The map of the scenario.

    """
    def __init__(self, map_, participants, max_step):
        self.map_ = map_
        self.participants = participants
        self.max_step = max_step

    def check_collision(self, mode):
        if mode == "participant":
            return self._check_collision_participant()
        elif mode == "environment":
            return self._check_collision_environment()
        
    def check_retrograde(self):
        return TrafficEvent.VIOLATION_RETROGRADE
    
    def check_non_drivable(self):
        return TrafficEvent.VIOLATION_NON_DRIVABLE
    
    def check_outbound(self):
        return TrafficEvent.OUTSIDE_MAP
    
    def check_time_exceed(self, curr_step):
        if curr_step < self.max_step:
            return TrafficEvent.NORMAL
        return TrafficEvent.TIME_EXCEED

    def check_complete(self):
        return TrafficEvent.ROUTE_COMPLETED

    def check_status(self):
        return