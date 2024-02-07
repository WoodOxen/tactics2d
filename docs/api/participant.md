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

The definition of vehicle types is based on the [European Emissions Standards](https://en.wikipedia.org/wiki/Vehicle_size_class#EEC) (EEC) due to its clarity and simplicity. To obtain the parameters, we choose one specific vehicle from each type of vehicle based on typical (highest selling) vehicles and available data found online. These choices were made to ensure the data used is as representative and accurate as possible.

The default vehicle parameters can be visited by calling [`tactics2d.participant.element.list_vehicle_templates()`](#tactics2d.participant.element.list_vehicle_templates).

| EEE Category | Prototype | Length (m) | Width (m) | Height (m) | Wheelbase (m) | Front overhang (m) | Rear overhang (m) | Kerb Weight (kg) | Max speed (m/s) | 0-100 km/h (s) | Driven mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: Mini car | [Volkswagen Up](https://en.wikipedia.org/wiki/Volkswagen_Up) | 3.540 | 1.641 | 1.489 | 2.420 | 0.585 | 0.535 | 1070 | 44.444 | 14.4 | FWD |
| B: Small car | [Volkswagen Polo](https://en.wikipedia.org/wiki/Volkswagen_Polo) | 4.053 | 1.751 | 1.461 | 2.548 | 0.824 | 0.681 | 1565 | 52.778 | 8.2 | FWD |
| C: Medium car | [Volkswagen Golf](https://en.wikipedia.org/wiki/Volkswagen_Golf_Mk8) | 4.284 | 1.799 | 1.452 | 2.637 | 0.880 | 0.767 | 1620 | 69.444 | 5.4 | FWD |
| D: Large car | [Volkswagen Passat](https://en.wikipedia.org/wiki/Volkswagen_Passat_(B8)) | 4.866 | 1.832 | 1.477 | 2.871 | 0.955 | 1.040 | 1735 | 58.333 | 8.4 | FWD |
| E: Executive car | [Audi A6](https://en.wikipedia.org/wiki/Audi_A6) | 5.050 | 1.886 | 1.475 | 3.024 | 0.921 | 1.105  | 2175 | 63.889 | 6.7 | FWD |
| F: Luxury car | [Audi A8](https://en.wikipedia.org/wiki/Audi_A8#Fourth_generation_(D5;_2018%E2%80%93present)) | 5.302 | 1.945 | 1.488 | 3.128 | 0.989 | 1.185 | 2520 | 69.444 | 6.7 | AWD |
| J: SUV | 
| M: MPV | [Volkswagen Sharan](https://en.wikipedia.org/wiki/Volkswagen_Sharan) | 4.855 | 1.904 | 1.720 | 2.920 | 0.968 | 0.967 | 1923 | 55.556 | 9.9 | AWD |
| S: Sports car | 
