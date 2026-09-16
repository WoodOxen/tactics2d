# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Vehicle implementation."""


import logging
from typing import Any, Tuple

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, LineString

from tactics2d.participant.trajectory import State, Trajectory
from tactics2d.physics import SingleTrackDrift, SingleTrackDynamics, SingleTrackKinematics

from .participant_base import ParticipantBase
from .participant_template import EPA_MAPPING, EURO_SEGMENT_MAPPING, NCAP_MAPPING, VEHICLE_TEMPLATE


class Vehicle(ParticipantBase):
    r"""This class defines a four-wheeled vehicle with its common properties.

    The definition of different driven modes of the vehicles can be found here:

    - [Front-Wheel Drive (FWD)](https://en.wikipedia.org/wiki/Front-wheel_drive)
    - [Rear-Wheel Drive (RWD)](https://en.wikipedia.org/wiki/Rear-wheel_drive)
    - [Four-Wheel Drive (4WD)](https://en.wikipedia.org/wiki/Four-wheel_drive)
    - [All-Wheel Drive (AWD)](https://en.wikipedia.org/wiki/All-wheel_drive)

    The default physics model provided for the vehicles is a kinematics bicycle model with the vehicle's geometry center as the origin. This model is recommended only for the vehicles with front-wheel drive (FWD). For other driven modes, the users should better custom their own physics model.

    !!! info "TODO"
        More physics models will be added in the future based on issues and requirements. You are welcome to suggest implementation of other physics models [here](https://github.com/WoodOxen/tactics2d/issues).

    Attributes:
        id_ (int): The unique identifier of the vehicle.
        type_ (str): The type of the vehicle. Defaults to "medium_car".
        trajectory (Trajectory): The trajectory of the vehicle. Defaults to an empty trajectory.
        color (tuple): The color of the vehicle. The color of the traffic participant. This attribute will be left to the sensor module to verify and convert to the appropriate type. You can refer to [Matplotlib's way](https://matplotlib.org/stable/users/explain/colors/colors.html) to specify validate colors. Defaults to light-turquoise "#2bcbba".
        length (float): The length of the vehicle. The unit is meter. Defaults to None.
        width (float): The width of the vehicle. The unit is meter. Defaults to None.
        height (float): The height of the vehicle. The unit is meter. Defaults to None.
        kerb_weight: (float): The weight of the vehicle. The unit is kilogram (kg). Defaults to None.
        wheel_base (float): The wheel base of the vehicle. The unit is meter. Defaults to None.
        front_overhang (float): The front overhang of the vehicle. The unit is meter. Defaults to None.
        rear_overhang (float): The rear overhang of the vehicle. The unit is meter. Defaults to None.
        driven_mode: (str): The driven way of the vehicle. The available options are ["FWD", "RWD", "4WD", "AWD"]. Defaults to "FWD".
        max_steer (float): The maximum approach angle of the vehicle. The unit is radian. Defaults to $\pi$/6.
        max_speed (float): The maximum speed of the vehicle. The unit is meter per second. Defaults to 55.56 (= 200 km/h).
        max_accel (float): The maximum acceleration of the vehicle. The unit is meter per second squared. Defaults to 3.0.
        max_decel (float): The maximum deceleration of the vehicle. The unit is meter per second squared. Defaults to 10.
        steer_range (Tuple[float, float]): The range of the vehicle steering angle. The unit is radian. Defaults to (-$\pi$/6, $\pi$/6).
        speed_range (Tuple[float, float]): The range of the vehicle speed. The unit is meter per second (m/s). Defaults to (-16.67, 55.56) (= -60~200 km/h).
        accel_range (Tuple[float, float]): The range of the vehicle acceleration. The unit is meter per second squared. Defaults to (-10, 3).
        verify (bool): Whether to verify the trajectory to bind or the state to add. Defaults to False.
        physics_model (PhysicsModelBase): The physics model of the vehicle. Defaults to SingleTrackKinematics.
        geometry (LinearRing): The geometry shape of the vehicle. It is represented as a bounding box with its original point located at the center. This attribute is **read-only**.
        current_state (State): The current state of the traffic participant. This attribute is **read-only**.
    """

    __annotations__ = {
        "type_": str,
        "length": float,
        "width": float,
        "height": float,
        "kerb_weight": float,
        "wheel_base": float,
        "front_overhang": float,
        "rear_overhang": float,
        "driven_mode": str,
        "max_steer": float,
        "max_speed": float,
        "max_accel": float,
        "max_decel": float,
        "verify": bool,
    }
    _default_color = "#2bcbba"  # light-turquoise
    _driven_modes = {"FWD", "RWD", "4WD", "AWD"}

    def __init__(
        self, id_: Any, type_: str = "medium_car", trajectory: Trajectory = None, **kwargs
    ):
        r"""Initialize the vehicle.

        Args:
            id_ (int): The unique identifier of the vehicle.
            type_ (str, optional): The type of the vehicle.
            trajectory (Trajectory, optional): The trajectory of the vehicle.

        Keyword Args:
            length (float, optional): The length of the vehicle. The unit is meter. Defaults to None.
            width (float, optional): The width of the vehicle. The unit is meter. Defaults to None.
            height (float, optional): The height of the vehicle. The unit is meter. Defaults to None.
            color (tuple, optional): The color of the vehicle. Defaults to None.
            kerb_weight (float, optional): The kerb weight of the vehicle. The unit is kilogram (kg). Defaults to None.
            wheel_base (float, optional): The wheel base of the vehicle. The unit is meter. Defaults to None.
            front_overhang (float, optional): The front overhang of the vehicle. The unit is meter. Defaults to None.
            rear_overhang (float, optional): The rear overhang of the vehicle. The unit is meter. Defaults to None.
            max_steer (float, optional): The maximum steering angle of the vehicle. The unit is radian. Defaults to $\pi$ / 6.
            max_speed (float, optional): The maximum speed of the vehicle. The unit is meter per second. Defaults to 55.56 (=200 km/h).
            max_accel (float, optional): The maximum acceleration of the vehicle. The unit is meter per second squared. Defaults to 3.0.
            max_decel (float, optional): The maximum deceleration of the vehicle. The unit is meter per second squared. Defaults to 10.0.
            verify (bool): Whether to verify the trajectory to bind or the state to add.
            physics_model (PhysicsModelBase): The physics model of the cyclist. Defaults to None. If the physics model is a custom model, it should be an instance of the [`PhysicsModelBase`](../api/physics.md/#PhysicsModelBase) class.
            driven_mode (str, optional): The driven way of the vehicle. The available options are ["FWD", "RWD", "4WD", "AWD"]. Defaults to "FWD".
        """

        super().__init__(id_, type_, trajectory, **kwargs)

        self._pose_cache = None  # (frame, pose) cached per frame; see get_pose

        self.max_steer = np.round(np.pi / 6, 3) if self.max_steer is None else self.max_steer
        self.max_speed = 55.56 if self.max_speed is None else self.max_speed
        self.max_accel = 3.0 if self.max_accel is None else self.max_accel
        self.max_decel = 10.0 if self.max_decel is None else self.max_decel

        self.speed_range = (-16.67, self.max_speed)
        self.steer_range = (-self.max_steer, self.max_steer)
        self.accel_range = (-self.max_accel, self.max_accel)

        if self.driven_mode is None:
            self.driven_mode = "FWD"
        elif self.driven_mode not in self._driven_modes:
            self.driven_mode = "FWD"
            logging.warning("Invalid driven mode. The default mode FWD will be used.")

        if self.verify:
            if "physics_model" in kwargs and kwargs["physics_model"] is not None:
                self.physics_model = kwargs["physics_model"]
            else:
                self._auto_construct_physics_model()

        if None not in [self.length, self.width]:
            self._bbox = LinearRing(
                [
                    [0.5 * self.length, -0.5 * self.width],
                    [0.5 * self.length, 0.5 * self.width],
                    [-0.5 * self.length, 0.5 * self.width],
                    [-0.5 * self.length, -0.5 * self.width],
                ]
            )
        else:
            self._bbox = None

    @property
    def geometry(self) -> LinearRing:
        return self._bbox

    def build_physics_model(
        self,
        model_class=SingleTrackKinematics,
        *,
        mu: float = None,
        I_z: float = None,
        cf: float = None,
        cr: float = None,
        radius: float = None,
        mass_height: float = None,
        interval: int = 100,
        delta_t: int = None,
        **kwargs,
    ):
        """Build a physics model from this vehicle's template parameters.

        The real geometry and mass of the participant (``lf``/``lr`` from the overhangs or
        half length, ``mass`` from the kerb weight, ``mass_height`` from half the height) and
        its steer/speed/accel ranges are filled in automatically, so only vehicle/road
        specific knobs need to be passed explicitly. This method does not assign
        ``self.physics_model``; callers that want verification can assign the return value.

        Note that the vehicle template carries geometry, mass and performance only - it has no
        real ``I_z`` (not published for production cars), so dynamics/drift keep the
        CommonRoad reference-vehicle defaults (Ford Escort) unless overridden. ``mu`` is a
        tire-road property (not a vehicle property): use the same value for every model that
        shares a road surface.

        Args:
            model_class (type, optional): A physics model class, one of
                ``SingleTrackKinematics`` (default), ``SingleTrackDynamics`` or
                ``SingleTrackDrift``.
            mu (float, optional): Tire-road friction coefficient. Applied to the kinematics
                (grip-limit cap) and the dynamics (friction coefficient) models. ``SingleTrackDrift``
                derives friction from its Pacejka tire, so ``mu`` is ignored for it.
            I_z (float, optional): Yaw moment of inertia (kg/m^2). None keeps the class default.
            cf (float, optional): Front cornering stiffness (1/rad). Dynamics only; None keeps the class default.
            cr (float, optional): Rear cornering stiffness (1/rad). Dynamics only; None keeps the class default.
            radius (float, optional): Effective wheel radius (m). Drift only; None keeps the class default.
            mass_height (float, optional): Center-of-mass height (m). Defaults to half the vehicle height.
            interval (int, optional): Physics step interval (ms). Defaults to 100.
            delta_t (int, optional): Internal sub-step (ms). None uses the class default.

        Keyword Args:
            **kwargs: Extra keyword arguments passed to the model constructor, e.g.
                ``friction_ellipse=True`` for ``SingleTrackDynamics``.

        Returns:
            physics_model (PhysicsModelBase): The constructed physics model.

        Raises:
            ValueError: If the model requires ``mass_height`` but the vehicle has no height
                and none was provided.
        """
        if self.length is None:
            raise ValueError("The vehicle has no length; cannot derive the wheel base.")
        if None not in [self.front_overhang, self.rear_overhang]:
            lf = self.length / 2 - self.front_overhang
            lr = self.length / 2 - self.rear_overhang
        else:
            lf = self.length / 2
            lr = self.length / 2

        params = dict(
            lf=lf,
            lr=lr,
            steer_range=self.steer_range,
            speed_range=self.speed_range,
            accel_range=self.accel_range,
            interval=interval,
            delta_t=delta_t,
            **kwargs,
        )

        needs_mass = issubclass(model_class, (SingleTrackDynamics, SingleTrackDrift))
        if needs_mass:
            params["mass"] = self.kerb_weight
            if mass_height is None:
                if self.height is None:
                    raise ValueError(
                        "The model needs mass_height but the vehicle has no height; pass mass_height explicitly."
                    )
                mass_height = self.height / 2
            params["mass_height"] = mass_height

        if issubclass(model_class, SingleTrackDynamics):
            if mu is not None:
                params["mu"] = mu
            if I_z is not None:
                params["I_z"] = I_z
            if cf is not None:
                params["cf"] = cf
            if cr is not None:
                params["cr"] = cr
        elif issubclass(model_class, SingleTrackDrift):
            if I_z is not None:
                params["I_z"] = I_z
            if radius is not None:
                params["radius"] = radius
        elif mu is not None:
            # SingleTrackKinematics: grip-limit cap is opt-in via mu.
            params["mu"] = mu

        return model_class(**params)

    def _auto_construct_physics_model(self):
        # Auto-construct the physics model for front-wheel-drive (FWD) vehicles.
        if self.driven_mode == "FWD":
            if self.length is not None:
                self.physics_model = self.build_physics_model(SingleTrackKinematics)
            else:
                self.verify = False
                logging.info(
                    "Cannot construct a physics model for the vehicle. The state verification is turned off."
                )
        # Auto-construct the physics model for rear-wheel-drive (RWD) vehicles.
        elif self.driven_mode == "RWD":
            pass
        # Auto-construct the physics model for all-wheel-drive (AWD) vehicles.
        else:
            pass

    def load_from_template(self, type_name: str, overwrite: bool = True, template: dict = None):
        """Load the vehicle properties from the template.

        Args:
            type_name (str): The type of the vehicle.
            overwrite (bool, optional): Whether to overwrite the existing properties. Defaults to False.
            template (dict, optional): The template of the vehicle. Defaults to VEHICLE_TEMPLATE.
        """
        if template is None:
            template = VEHICLE_TEMPLATE

        if type_name in EURO_SEGMENT_MAPPING:
            type_name = EURO_SEGMENT_MAPPING[type_name]
        elif type_name in EPA_MAPPING:
            type_name = EPA_MAPPING[type_name]
        elif type_name in NCAP_MAPPING:
            type_name = NCAP_MAPPING[type_name]

        if type_name in template:
            for key, value in template[type_name].items():
                if key == "0_100_km/h":
                    if overwrite or self.max_accel is None:
                        self.max_accel = np.round(100 * 1000 / 3600 / template[type_name][key], 3)
                else:
                    if overwrite or getattr(self, key) is None:
                        setattr(self, key, value)
        else:
            logging.warning(
                f"{type_name} is not in the vehicle template. The default values will be used."
            )

        self.speed_range = (-16.67, self.max_speed)
        self.accel_range = (-self.max_decel, self.max_accel)

        if None not in [self.length, self.width]:
            self._bbox = LinearRing(
                [
                    [0.5 * self.length, -0.5 * self.width],
                    [0.5 * self.length, 0.5 * self.width],
                    [-0.5 * self.length, 0.5 * self.width],
                    [-0.5 * self.length, -0.5 * self.width],
                ]
            )
        self._pose_cache = None

    def reset(self, state: State = None, keep_history: bool = False):
        """Reset the vehicle and invalidate its cached pose.

        Args:
            state (State, optional): The initial state of the object.
            keep_history (bool, optional): Whether to keep the record of history trajectory.
        """
        super().reset(state, keep_history)
        self._pose_cache = None

    def add_state(self, state: State):
        """This function adds a state to the vehicle.

        Args:
            state (State): The state to add.
        """
        if not self.verify or self.physics_model is None:
            self.trajectory.add_state(state)
        elif self.physics_model.verify_state(state, self.trajectory.current_state):
            self.trajectory.append_state(state)
        else:
            raise RuntimeError(
                "Invalid state checked by the physics model %s."
                % (self.physics_model.__class__.__name__)
            )
        self._pose_cache = None

    def bind_trajectory(self, trajectory: Trajectory):
        """This function binds a trajectory to the vehicle.

        Args:
            trajectory (Trajectory): The trajectory to bind.

        Raises:
            TypeError: If the input trajectory is not of type [`Trajectory`](#tactics2d.participant.trajectory.Trajectory).
        """
        if not isinstance(trajectory, Trajectory):
            raise TypeError("The trajectory must be an instance of Trajectory.")

        if self.verify:
            if not self._verify_trajectory(trajectory):
                self.trajectory = Trajectory(self.id_)
                logging.warning(
                    f"The trajectory is invalid. Vehicle {self.id_} is not bound to the trajectory."
                )
            else:
                self.trajectory = trajectory
        else:
            self.trajectory = trajectory
            logging.debug(f"Vehicle {self.id_} is bound to a trajectory without verification.")
        self._pose_cache = None

    def get_pose(self, frame: int = None) -> LinearRing:
        """This function gets the pose of the vehicle at the requested frame.

        The pose is cached per frame and invalidated whenever the trajectory is
        mutated (``add_state`` / ``bind_trajectory`` / ``reset``) or the bounding
        box is rebuilt (``load_from_template``), so repeated queries within one
        simulation step reuse the same Shapely ``affine_transform``.

        Args:
            frame (int, optional): The frame to get the vehicle's pose.

        Returns:
            pose (LinearRing): The vehicle's bounding box which is rotated and moved based on the current state.
        """
        if self._pose_cache is not None and self._pose_cache[0] == frame:
            return self._pose_cache[1]

        state = self.trajectory.get_state(frame)
        transform_matrix = [
            np.cos(state.heading),
            -np.sin(state.heading),
            np.sin(state.heading),
            np.cos(state.heading),
            state.location[0],
            state.location[1],
        ]
        pose = affine_transform(self._bbox, transform_matrix)
        self._pose_cache = (frame, pose)
        return pose

    def get_trace(self, frame_range: Tuple[int, int] = None) -> LinearRing:
        """This function gets the trace of the vehicle within the requested frame range.

        Args:
            frame_range (Tuple[int, int], optional): The requested frame range. The first element is the start frame, and the second element is the end frame. The unit is millisecond (ms).

        Returns:
            LinearRing: The trace of the vehicle within the requested frame range, represented as a polygon buffer around the trajectory centerline.
        """

        trace_points = self.trajectory.get_trace(frame_range)
        if not trace_points:
            # Return empty LinearRing if no points
            return LinearRing()

        center_line = LineString(trace_points)
        if self.width is not None:
            buffer = center_line.buffer(self.width / 2, cap_style="square")
            return LinearRing(buffer.exterior.coords)
        else:
            # If width is not available, return the center line as a closed ring
            # by duplicating the first point at the end
            coords = list(center_line.coords)
            if len(coords) >= 2:
                coords.append(coords[0])
            return LinearRing(coords)
