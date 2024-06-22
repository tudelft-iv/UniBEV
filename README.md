# UniBEV: Multi-modal 3D Object Detection with Uniform BEV Encoders for Robustness against Missing Sensor Modalities
Shiming Wang, Holger Caesar, Liangliang Nan, and Julian F. P. Kooij

The official implementation of the IVS'24 paper UniBEV. [Paper in arxiv](https://arxiv.org/abs/2309.14516)

## Abstract
Multi-sensor object detection is an active research topic in automated driving, but the robustness of such detection models against missing sensor input (modality missing), e.g., due to a sudden sensor failure, is a critical problem which remains under-studied. In this work, we propose UniBEV, an end-to-end multi-modal 3D object detection framework designed for robustness against missing modalities:UniBEV can operate on LiDAR plus camera input, but also on LiDAR-only or camera-only input without retraining.
To facilitate its detector head to handle different input combinations,  UniBEV aims to create well-aligned Bird's Eye View (BEV) feature maps from each available modality.
Unlike prior BEV-based multi-modal detection methods,
all sensor modalities follow a uniform approach to resample features from the native sensor coordinate systems to the BEV features. We furthermore investigate the robustness of various fusion strategies w.r.t. missing modalities: the commonly used feature concatenation, but also channel-wise averaging, and a generalization to weighted averaging termed Channel Normalized Weights. To validate its effectiveness, we compare UniBEV to state-of-the-art BEVFusion and MetaBEV on nuScenes over all sensor input combinations. In this setting, UniBEV achieves $52.5 \%$ mAP on average over all input combinations, significantly improving over the baselines
($43.5 \%$ mAP on average for BEVFusion, $48.7 \%$ mAP on average for MetaBEV). An ablation study shows the robustness benefits of fusing by weighted averaging over regular concatenation, of selecting  reasonable probabilities of different modalities as input during training, and of sharing queries between the BEV encoders of each modality.

## Update Logs
+ 2024.05.30: updated the core codes and components
+ 2024.06.22: updated the config files, pre-trained weights for the main results
+ 2024.06.22: upload the Singularity image and recipe

## TODO List
+ upload the pre-trained weights for the ablation studies
+ upload inference configuration files

## Methods
![UniBEV](/assets/unibev.png)

## Getting Started
- Set Environment with cuda 11.3 (Singularity [image](https://surfdrive.surf.nl/files/index.php/s/UMfSSSb5T40kcwd), [recipe](https://surfdrive.surf.nl/files/index.php/s/37HNgaVGEWAowet))
- [Installation](docs/installation.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Eval](docs/run_eval.md)
## Models on nuScenes val
### Pretrained weights
model
### Main results
|Method|Training Modality|L+C mAP| L mAP| C mAP|Summary| Model |
|------|:---------------:|:-----:|------|------|-------|-------|
|[UniBEV_C](/projects/UniBEV/configs/unibev/unibev_nus_C.py)|C|-|-|36.9|-|link|
|[UniBEV_L](/projects/UniBEV/configs/unibev/unibev_nus_L.py)|L|-|57.8|-|-|link|
|[UniBEV_CNW](/projects/UniBEV/configs/unibev/unibev_nus_LC_cnw_256_modality_dropout.py)|L+C(MD)|64.2|58.2|35.0|52.5|[link](https://surfdrive.surf.nl/files/index.php/s/CX1xt8FeUiiDlIS)|
|[UniBEV_avg](/projects/UniBEV/configs/unibev/unibev_nus_LC_avg_256_modality_dropout.py)|L+C(MD)|64.1|57.6|35.1|52.3|[link](https://surfdrive.surf.nl/files/index.php/s/QH2N9EJlPM2oaiT)|       
|[UniBEV_cat](/projects/UniBEV/configs/unibev/unibev_nus_LC_cat_128_modality_dropout.py)|L+C(MD)|63.8|57.6|34.4|51.9|[link](https://surfdrive.surf.nl/files/index.php/s/w8nhdpUPhrrkus8)|

Please refer the paper for more details.

## Ablations on Modality Dropout strategy 
### Effect of Sensor Dropping Probabilities p<sub>L</sub> and p<sub>C</sub>
|Model|p<sub>md</sub>|p<sub>L</sub>|p<sub>C</sub>|L+C mAP|L mAP|C mAP|Summary| Model|
|-|-|-|-|:--:|:--:|:--:|:--:|:--:|
|[UniBEV_CNW](/projects/UniBEV/configs/unibev/ablation_md/unibev_nus_LC_cnw_256_modality_dropout_m50s50l0c100.py)|0.5|0   |1   |63.2|45.5|36.0|48.2|link|
|[UniBEV_CNW](/projects/UniBEV/configs/unibev/ablation_md/unibev_nus_LC_cnw_256_modality_dropout_m50s50l25c75.py)|0.5|0.25|0.75|64.0|57.8|35.8|52.5|link|
|_**UniBEV_CNW**_|0.5|0.5 |0.5|64.2|58.2|35.0|52.5|link|
|[UniBEV_CNW](/projects/UniBEV/configs/unibev/ablation_md/unibev_nus_LC_cnw_256_modality_dropout_m50s50l75c25.py)|0.5|0.75|0.25|63.8|58.3|33.2|51.8|link|
|[UniBEV_CNW](/projects/UniBEV/configs/unibev/ablation_md/unibev_nus_LC_cnw_256_modality_dropout_m50s50l0c100.py)|0.5|0   |1   |60.8|55.9|3.0 |39.9|link|

### Effect of Modality Dropout Probability p<sub>md</sub> (not included in the paper)
|Model|p<sub>md</sub>|p<sub>L</sub>|p<sub>C</sub>|L+C mAP|L mAP|C mAP|Summary| Model|
|-|-|-|-|:--:|:--:|:--:|:--:|:--:|
|[UniBEV_CNW](/projects/UniBEV/configs/unibev/ablation_md/unibev_nus_LC_cnw_256_modality_dropout_m0s100l50c50.py)|1|0.5|0.5|60.9|56.5|37.1|51.5|link|
|[UniBEV_CNW](/projects/UniBEV/configs/unibev/ablation_md/unibev_nus_LC_cnw_256_modality_dropout_m25s75l50c50.py)|0.75|0.5|0.5|63.1|56.2|36.6|52.0|link|
|_**UniBEV_CNW**_|0.5|0.5 |0.5|64.2|58.2|35.0|52.5|link|
|[UniBEV_CNW](/projects/UniBEV/configs/unibev/ablation_md/unibev_nus_LC_cnw_256_modality_dropout_m75s25l50c50.py)|0.25|0.5|0.5|62.6|55.9|33.6|50.7|link|
|[UniBEV_CNW](/projects/UniBEV/configs/unibev/ablation_md/unibev_nus_LC_cnw_256_modality_dropout_m100s0l50c50.py)|0|0.5|0.5|62.6|50.6|4.6 |39.9|link|


## Poster
![Poster at IV Symposium'24](/assets/UniBEV_poster_IV24.png)
[Download Poster via here](https://surfdrive.surf.nl/files/index.php/s/Kuxogt4IKdPuNgz)
## Citation
```
@inproceedings{wang2023unibev,
  title={UniBEV: Multi-modal 3D Object Detection with Uniform BEV Encoders for Robustness against Missing Sensor Modalities},
  author={Wang, Shiming and Caesar, Holger and Nan, Liangliang and Kooij, Julian FP},
  booktitle={2024 IEEE Intelligent Vehicles Symposium (IV)},
  year={2024}
}
```

## Acknowledgement
