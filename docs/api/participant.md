::: tactics2d.participant

::: tactics2d.participant.trajectory
    options:
        heading_level: 2
        members:
            - State
            - Trajectory

::: tactics2d.participant.element
    options:
        heading_level: 2
        members:
            - ParticipantBase
            - Vehicle
            - Cyclist
            - Pedestrian
            - Other
            - list_vehicle_templates
            - list_cyclist_templates
            - list_pedestrian_templates

## Templates for Traffic Participants

### Four-wheel Vehicle Models

Our classification of vehicle types adheres to the [European Emissions Standards](https://en.wikipedia.org/wiki/Vehicle_size_class#EEC) (EEC) due to its clear and straightforward categorization. To establish the parameters, we select a specific vehicle representing each type based on commonly sold models and accessible data sourced online. These selections are carefully made to ensure the data utilized is both representative and accurate.

Due to the challenge of obtaining precise maximum steering values for each vehicle type, we assume a uniform maximum steering value of $\pi/6$ radians for all vehicles. This assumption is based on the understanding that, as our vehicle physics model operates on the bicycle model, subtle variations in steering range are unlikely to significantly impact the simulation's outcomes.

The default vehicle parameters can be visited by calling [`tactics2d.participant.element.list_vehicle_templates()`](#tactics2d.participant.element.list_vehicle_templates).

| EEE Category | Prototype | Length (m) | Width (m) | Height (m) | Wheelbase (m) | Front overhang (m) | Rear overhang (m) | Kerb Weight (kg) | Max speed (m/s) | 0-100 km/h (s) | Driven mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: Mini car | [Volkswagen Up](https://en.wikipedia.org/wiki/Volkswagen_Up) | 3.540 | 1.641 | 1.489 | 2.420 | 0.585 | 0.535 | 1070 | 44.44 | 14.4 | FWD |
| B: Small car | [Volkswagen Polo](https://en.wikipedia.org/wiki/Volkswagen_Polo) | 4.053 | 1.751 | 1.461 | 2.548 | 0.824 | 0.681 | 1565 | 52.78 | 11.2 | FWD |
| C: Medium car | [Volkswagen Golf](https://en.wikipedia.org/wiki/Volkswagen_Golf_Mk8) | 4.284 | 1.799 | 1.452 | 2.637 | 0.880 | 0.767 | 1620 | 69.44 | 8.9 | FWD |
| D: Large car | [Volkswagen Passat](https://en.wikipedia.org/wiki/Volkswagen_Passat_(B8)) | 4.866 | 1.832 | 1.477 | 2.871 | 0.955 | 1.040 | 1735 | 58.33 | 8.4 | FWD |
| E: Executive car | [Audi A6](https://en.wikipedia.org/wiki/Audi_A6) | 5.050 | 1.886 | 1.475 | 3.024 | 0.921 | 1.105  | 2175 | 63.89 | 8.1 | FWD |
| F: Luxury car | [Audi A8](https://en.wikipedia.org/wiki/Audi_A8#Fourth_generation_(D5;_2018%E2%80%93present)) | 5.302 | 1.945 | 1.488 | 3.128 | 0.989 | 1.185 | 2520 | 69.44 | 6.7 | AWD |
| S: Sports coupe | [Ford Mustang](https://en.wikipedia.org/wiki/Ford_Mustang) | 4.788 | 1.916 | 1.381 | 2.720 | 0.830 | 1.238 | 1740 | 63.89 | 5.3 | AWD |
| M: MPV | [Kia Carnival](https://en.wikipedia.org/wiki/Kia_Carnival) | 5.155 | 1.995 | 1.740 | 3.090 | 0.935 | 1.130 | 2095 | 66.67 | 9.4 | 4WD |
| J: SUV | [Jeep Grand Cherokee](https://en.wikipedia.org/wiki/Jeep_Grand_Cherokee) | 4.828 | 1.943 | 1.792 | 2.915 | 0.959 | 0.954 | 2200 | 88.89 | 3.8 | 4WD |

### Cyclist Models

The cyclist model is designed around average parameters. To access the default cyclist parameters, you can call [`tactics2d.participant.element.list_cyclist_templates()`](#tactics2d.participant.element.list_cyclist_templates).

| Name | Length (m) | Width (m) | Height (m) | Max steer (rad) | Max speed (m/s) | Max acceleration (m/s$^2$) | Max deceleration (m/s$^2$) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cyclist | 1.80 | 0.65 | 1.70 | 1.05 | 22.78 | 5.8 | 7.8 |
| Moped | 2.00 | 0.70 | 1.70 | 0.35 | 13.89 | 3.5 | 7.0 |
| Motorcycle | 2.40 | 0.80 | 1.70 | 0.44 | 75.00 | 5.0 | 10.0 |

### Pedestrian Models

The pedestrian model is designed around average parameters. To access the default pedestrian parameters, you can call [`tactics2d.participant.element.list_pedestrian_templates()`](#tactics2d.participant.element.list_pedestrian_templates).

| Name | Length (m) | Width (m) | Height (m) | Max speed (m/s) | Max acceleration (m/s$^2$) |
| --- | --- | --- | --- | --- | --- |
| Adult/male | 0.24 | 0.40 | 1.75 | 7.0 | 1.5 |
| Adult/female | 0.22 | 0.37 | 1.65 | 6.0 | 1.5 |
| Child (six-year old) | 0.18 | 0.25 | 1.16 | 3.5 | 1.0 |
| Child (ten-year old) | 0.20 | 0.35 | 1.42 | 4.5 | 1.0 |
