from abc import ABC, abstractmethod

from tactics2d.trajectory.element.trajectory import State, Trajectory


class ParticipantBase(ABC):
    """This class defines an interface for all the traffic participants provided in tactics2d.

    Please feel free to inherit this class to define their own traffic participants.

    Attributes:
        id_ (int): The unique identifier of the traffic participant.
        type_ (str): The type of the traffic participant. Defaults to None.
        length (float): The length of the traffic participant. The default unit is meter (m).
            Defaults to None.
        width (float): The width of the traffic participant. The default unit is meter (m).
            Defaults to None.
        height (float): The height of the traffic participant. The default unit is meter (m).
            Defaults to None.
        trajectory (Trajectory): The trajectory of the traffic participant. Defaults to None.
    """

    default_attributes = {
        "length": float,
        "width": float,
        "height": float,
    }

    def __init__(
        self,
        id_: int,
        type_: str,
        **kwargs,
    ):
        """The basic constructor of the traffic participant.

        By defaults the traffic participant has the properties id_, type_, length, width, and
        height.

        If you need to customize more attributes, you can define them in the inherent
        class in the dictionary ```attributes```. The key of the dictionary is the name of
        the attribute, and the value of the dictionary is the type of the attribute. If the
        type of the custom attribute is not determined, the corresponding dictionary value
        should be ```None```. The type of the attributes will be checked in the construction.
        If the type is not correct, the constructor will try to convert the value to the correct
        type. If the conversion fails, the constructor will assign ```None``` to the attribute.

        All the attributes defined in the dictionaries ```default_attributes``` and ```attributes```
        will be initialized. If their values are not specified in the constructor, they will be
        initialized as ```None```.

        If ```**kwargs``` contains any key that is not defined in the dictionaries ```default_attributes```
        and ```attributes```, the constructor will assign the value of the key to the attribute with the
        same name.
        """
        self.id_ = id_
        self.type_ = type_

        attribute_dict = (
            self.default_attributes
            if not hasattr(self, "attributes")
            else {**self.default_attributes, **self.attributes}
        )

        for key, value in attribute_dict.items():
            if key in kwargs:
                if value is None or isinstance(kwargs[key], value):
                    setattr(self, key, kwargs[key])
                else:
                    try:
                        setattr(self, key, value(kwargs[key]))
                    except:
                        setattr(self, key, None)
            else:
                setattr(self, key, None)

        for key in kwargs.keys():
            if key not in attribute_dict:
                setattr(self, key, kwargs[key])

        self.trajectory = Trajectory(id_=self.id_)
        if kwargs.get("trajectory", None) is not None:
            self.bind_trajectory(kwargs["trajectory"])

    @property
    def current_state(self) -> State:
        return self.trajectory.get_state()

    @property
    def location(self):
        return self.current_state.location

    @property
    def heading(self) -> float:
        return self.current_state.heading

    @property
    def velocity(self):
        return self.current_state.velocity

    @property
    def speed(self) -> float:
        return self.current_state.speed

    @property
    def accel(self):
        return self.current_state.accel

    def is_active(self, frame: int) -> bool:
        """Check if the participant has state information at the requested frame."""
        if frame < self.trajectory.first_frame or frame > self.trajectory.last_frame:
            return False
        return True

    @abstractmethod
    def _verify_trajectory(self, trajectory: Trajectory):
        """Check if the trajectory is allowed by the participant's physical constraints.

        Returns:
            bool: True if the trajectory is valid, False otherwise.
        """

    @abstractmethod
    def bind_trajectory(self, trajectory: Trajectory):
        """Bind a trajectory with the traffic participant."""

    @abstractmethod
    def get_pose(self, frame: int = None):
        """Get the traffic participant's pose at the requested frame.

        If the frame is not specified, the function will return the current pose.
        If the frame is requested but not found, the function will raise a TrajectoryKeyError.
        """

    def get_state(self, frame: int = None):
        """Get the traffic participant's state at the requested frame.

        Args:
            frame (int, optional): The requested frame. If the frame is not specified, the
                function will return the current state. If the frame is requested but not found,
                the function will raise a KeyError.

        Returns:
            State: The traffic participant's state at the requested frame.

        Raises:
            KeyError: The requested frame is not found in the trajectory.
        """
        return self.trajectory.get_state(frame)

    def get_states(self, frame_range=None) -> list:
        """Get the traffic participant's states within the requested frame range.

        Args:
            frame_range (_type_, optional): The requested frame range. If the frame range is
                not specified, the function will return all states. If the frame range is a tuple,
                the function will return the states within the requested range. If the frame range
                is a list, the function will return the states at the requested frames. If the frame
                range is an element same as the frame id, the function will return a list only containing
                the state. Defaults to None.

        Returns:
            list: A list of the traffic participant's states.

        Raises:
            TypeError: The frame range must be a tuple, or a list, or an element same as the frame id.
            ValueError: The frame range must be a tuple with two elements.
            KeyError: Any requested frame is not found in the trajectory.
        """
        frames = self.trajectory.frames
        states = []
        if frame_range is None:
            for frame in frames:
                states.append(self.trajectory.get_state(frame))
        elif isinstance(frame_range, tuple):
            if len(frame_range) == 2:
                start_frame, end_frame = frame_range
                for frame in frames:
                    if frame >= start_frame and frame <= end_frame:
                        states.append(self.trajectory.get_state(frame))
            else:
                raise ValueError("The frame range must be a tuple with two elements.")
        elif isinstance(frame_range, list):
            for frame in frame_range:
                states.append(self.trajectory.get_state(frame))
        else:
            try:
                states.append(self.trajectory.get_state(frame_range))
            except:
                raise TypeError(
                    "The frame range must be a tuple, or a list, or an element same as the frame id."
                )

        return states

    def get_trajectory(self, frame_range=None) -> list:
        """Get the traffic participant's trajectory within the requested frame range.

        Args:
            frame_range (_type_, optional): The requested frame range. If the frame range is
                not specified, the function will return the whole trajectory. If the frame range
                is a tuple, the function will return the trajectory within the requested range. If the
                frame range is a list, the function will return the trajectory at the requested frames.
                If the frame range is an element same as the frame id, the function will return
                a list only containing the location. Defaults to None.

        Returns:
            list: A list of the traffic participant's history locations.

        Raises:
            TypeError: The frame range must be a tuple, or a list, or an element same as the frame id.
            ValueError: The frame range must be a tuple with two elements.
            KeyError: Any requested frame is not found in the trajectory.
        """
        frames = self.trajectory.frames
        trajectory = []
        if frame_range is None:
            for frame in frames:
                trajectory.append(self.trajectory.get_location(frame).location)
        elif isinstance(frame_range, tuple):
            if len(frame_range) == 2:
                start_frame, end_frame = frame_range
                for frame in frames:
                    if frame >= start_frame and frame <= end_frame:
                        trajectory.append(self.trajectory.get_location(frame).location)
            else:
                raise ValueError("The frame range must be a tuple with two elements.")
        elif isinstance(frame_range, list):
            for frame in frame_range:
                trajectory.append(self.trajectory.get_location(frame).location)
        else:
            try:
                trajectory.append(self.trajectory.get_location(frame_range).location)
            except:
                raise TypeError(
                    "The frame range must be a tuple, or a list, or an element same as the frame id."
                )

        return trajectory

    @abstractmethod
    def get_trace(self, frame_range=None):
        """Get the region boundary that the traffic participant has occupied within the requested frame range.

        Args:
            frame_range (_type_, optional): The requested frame range. Defaults to None.
        """

    def reset(self, state: State = None, keep_trajectory: bool = False):
        """Reset the object to a requested state. If the initial state is not specified, the object
                will be reset to the same initial state as previous.

        Args:
            state (State, optional): The initial state of the object. Defaults to None.
            keep_trajectory (bool, optional): Whether to keep the record of history trajectory.
                This argument only works when the state is not specified. When the state is
                not None, the trajectory will be reset to the new state.
                Defaults to False.
        """
        self.trajectory.reset(state, keep_trajectory)
