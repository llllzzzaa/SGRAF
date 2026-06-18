# SGRAF

Semantic Grounding and Reliability-Aware Fusion for RGB-T Object Detection under Adverse Illumination

## Brief Introduction

This repository serves as an extension to the paper "Semantic Grounding and Reliability-Aware Fusion for RGB-T Object Detection under Adverse Illumination"
To facilitate the review process, we release the evaluation code, experimental results, and pre-trained model weights in this repository.

## 1. 🚀 Get Start

**0. Install**

```bash
conda create -n SGRAF python=3.8 -y
conda activate SGRAF

# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install mmcv tqdm matplotlib scikit-learn opencv-python numpy==1.24.1
```

**1. Download Datasets**  
Please download the FLIR and LLVIP datasets from their official websites. The annotation files and the required dataset directory structure are provided in this repository.

**2.Download Checkpoints**  
Download pre-train and fine-tune model checkpoints from [here](https://pan.baidu.com/s/1a4BTsTbTCZ6eTMBMoaUa_A), code: rree.

**3.evaluation**
```
sh scripts/test.sh
```

## 2. 🌹 Acknowledgments
Our code is heavily based on [MMDetection](https://github.com/open-mmlab/mmdetection) and [UniRGB-IR](https://github.com/PoTsui99/UniRGB-IR), thanks for their excellent work!
