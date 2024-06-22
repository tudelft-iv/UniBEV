# Dataset Preparation for nuScenes
Download nuScenes V1.0 full dataset data [HERE](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data). Prepare nuscenes data by running
```shell
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

For more details, you can follow the instruction of [mmdetection3d](https://mmdetection3d.readthedocs.io/en/v0.18.1/data_preparation.html)