# TriLite - Official PyTorch Implementation (CVPR 2026)

This repository provides the official PyTorch implementation of:

> **TriLite: Efficient Weakly Supervised Object Localization with Universal Visual Features and Tri-Region Disentanglement**  
> Accepted at CVPR 2026.

TriLite is a parameter-efficient WSOL framework built on frozen universal visual representations.  
Despite training fewer than 1M parameters, TriLite achieves new state-of-the-art localization performance across standard WSOL benchmarks.

---

## 🛠️ Environment & Dependencies

This project is developed and tested using **Python 3.11**.

### Required Python packages:

* `torch==2.5.1`
* `torchvision==0.20.1`
* `opencv-python==4.10.0.84`
* `pillow==11.0.0`
* `munch==2.5.0`
* `tensorboard`
* `tqdm`

> ⚠️ `torch` and `torchvision` are installed using the official CUDA 12.4 wheels. Adjust the CUDA version if needed to match your system configuration.

---

## 📦 Installation

To install all required packages, run the following command:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 && \
pip install opencv-python==4.10.0.84 pillow==11.0.0 munch==2.5.0 tensorboard tqdm
```

---

## 📂 Dataset Preparation

We follow the standard WSOL dataset setup as described in the [Clova AI WSOL Evaluation repository](https://github.com/clovaai/wsolevaluation). Please refer to their instructions for downloading and organizing datasets such as CUB-200-2011 and ImageNet-1K.

---

## 🚀 Training

All training hyperparameters are defined in YAML configuration files located in the `configs/` directory.

To train the model (e.g., on CUB-200-2011), run:

```bash
python main.py --config 'configs/CUB_config.yaml'
```

During training, logs are recorded to TensorBoard. To monitor training:

```bash
tensorboard --logdir runs/
```

---

## 📊 Evaluation

To evaluate a trained model, provide the config file and the corresponding checkpoint path:

```bash
python evaluate.py --config 'configs/CUB_config.yaml' --checkpoint 'checkpoint/CUB/best_combined.pth'
```

---

## 📌 Acknowledgements

Parts of this codebase are borrowed and adapted from:

* [Clova AI WSOL Evaluation repository](https://github.com/clovaai/wsolevaluation)
* [DINOv2 by Facebook Research](https://github.com/facebookresearch/dinov2)
