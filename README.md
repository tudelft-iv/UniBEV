# UniBEV: Multi-modal 3D Object Detection with Uniform BEV Encoders for Robustness against Missing Sensor Modalities
Shiming Wang, Holger Caesar, Liangliang Nan, and Julian F. P. Kooij

The official implementation of the paper UniBEV. [Paper in arxiv](https://arxiv.org/abs/2309.14516)

## Abstract
Multi-sensor object detection is an active research topic in automated driving, but the robustness of such detection models against missing sensor input (modality missing), e.g., due to a sudden sensor failure, is a critical problem which remains under-studied. In this work, we propose UniBEV, an end-to-end multi-modal 3D object detection framework designed for robustness against missing modalities:UniBEV can operate on LiDAR plus camera input, but also on LiDAR-only or camera-only input without retraining.
To facilitate its detector head to handle different input combinations,  UniBEV aims to create well-aligned Bird's Eye View (BEV) feature maps from each available modality.
Unlike prior BEV-based multi-modal detection methods,
all sensor modalities follow a uniform approach to resample features from the native sensor coordinate systems to the BEV features. We furthermore investigate the robustness of various fusion strategies w.r.t. missing modalities: the commonly used feature concatenation, but also channel-wise averaging, and a generalization to weighted averaging termed Channel Normalized Weights. To validate its effectiveness, we compare UniBEV to state-of-the-art BEVFusion and MetaBEV on nuScenes over all sensor input combinations. In this setting, UniBEV achieves $52.5 \%$ mAP on average over all input combinations, significantly improving over the baselines
($43.5 \%$ mAP on average for BEVFusion, $48.7 \%$ mAP on average for MetaBEV). An ablation study shows the robustness benefits of fusing by weighted averaging over regular concatenation, and of sharing queries between the BEV encoders of each modality.

## Methods
![UniBEV](/assets/unibev.png)

## Getting Started
- [Installation](docs/installation.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Eval](docs/run_eval.md)
## Model Zoo

## Citation
```
@article{wang2023unibev,
  title={UniBEV: Multi-modal 3D Object Detection with Uniform BEV Encoders for Robustness against Missing Sensor Modalities},
  author={Wang, Shiming and Caesar, Holger and Nan, Liangliang and Kooij, Julian FP},
  journal={arXiv preprint arXiv:2309.14516},
  year={2023}
}
```

## Acknowledgement
