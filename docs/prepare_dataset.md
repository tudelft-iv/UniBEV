# Dataset Preparation for nuScenes
Download nuScenes V1.0 full dataset data [HERE](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data). Prepare nuscenes data by running
```shell
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```