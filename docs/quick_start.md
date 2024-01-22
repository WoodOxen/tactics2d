# Quick Start

## Visualize datasets

```shell
python samples/visualize_dataset.py \
    --dataset highd \
    --file 01_tracks.csv \
    --folder ./tactics2d/data/trajectory_sample/highD/data

python samples/visualize_dataset.py \
    --dataset highd \
    --file 60_tracks.csv \
    --folder ./tactics2d/data/trajectory/highD/data

python samples/visualize_dataset.py \
    --dataset ind \
    --file 00_tracks.csv \
    --folder ./tactics2d/data/trajectory_sample/inD/data

python samples/visualize_dataset.py \
    --dataset round \
    --file 00_tracks.csv \
    --folder ./tactics2d/data/trajectory_sample/rounD/data

python samples/visualize_dataset.py \
    --dataset exid \
    --file 01_tracks.csv \
    --folder ./tactics2d/data/trajectory_sample/exiD/data

python samples/visualize_dataset.py \
    --dataset argoverse \
    --file scenario_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.parquet \
    --folder ./tactics2d/data/trajectory_sample/Argoverse/train/0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca \
    --map-file log_map_archive_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.json

python samples/visualize_dataset.py \
    --dataset argoverse \
    --file  scenario_0a3f4a4b-7c3c-4c29-ae6d-edcaee524113.parquet \
    --folder ./tactics2d/data/trajectory/Argoverse/val/0a3f4a4b-7c3c-4c29-ae6d-edcaee524113 \
    --map-file log_map_archive_0a3f4a4b-7c3c-4c29-ae6d-edcaee524113.json

python samples/visualize_dataset.py \
    --dataset dlp \
    --file DJI_0012_agents.json \
    --folder ./tactics2d/data/trajectory_sample/DLP

python samples/visualize_dataset.py \
    --dataset interaction \
    --file pedestrian_tracks_000.csv \
    --folder ./tactics2d/data/trajectory_sample/INTERACTION/recorded_trackfiles/DR_USA_Intersection_EP0 \
    --map-name DR_USA_Intersection_EP0

python samples/visualize_dataset.py \
    --dataset nuplan \
    --file 2021.08.26.18.24.36_veh-28_00578_00663.db \
    --folder ./tactics2d/data/trajectory_sample/NuPlan/data/cache/train_boston

python samples/visualize_dataset.py \
    --dataset womd \
    --file motion_data_one_scenario.tfrecord \
    --folder ./tactics2d/data/trajectory_sample/WOMD
```

## Train with preset environments

## Export visualization results
