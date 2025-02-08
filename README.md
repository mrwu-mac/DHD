# Toward Open-set Human Object Interaction Detection

[![python](./assets/python.svg)](https://www.python.org/) [![pytorch](./assets/pytorch.svg)](https://pytorch.org/get-started/locally/) [![pocket](./assets/pocket.svg)](https://github.com/fredzzhang/pocket) [![license](./assets/license.svg)](./LICENSE)

This repository contains the official PyTorch implementation for the paper [Toward Open-set Human Object Interaction Detection (AAAI2024)](https://ojs.aaai.org/index.php/AAAI/article/view/28422).


## Model Zoo
We provide weights for DHD models trained on HICO-DET. 

| Model | Dataset | Default Settings | DHD Weights | GroundingDINO Weights |
|:-:|:-:|:-:|:-:|:-:|
| DHD | HICO-DET | (`29.91`, `28.42`, `30.35`) | [weights](https://drive.google.com/file/d/1zEo8MMiiXmLfgs46AziAkdb6MjJQFk6M/view?usp=sharing) | [weights](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth) |


## Prerequisites
1. Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket). The recommended PyTorch version is 1.9.0.
Make sure the environment for Pocket is activated (`conda activate pocket`), and install the packaging library with `pip install packaging`. 

2. init [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [CLIP(from VIPLO)](https://github.com/Jeeseung-Park/CLIP.git).

3. Prepare the [HICO-DET dataset](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk).
    1. If you have not downloaded the dataset before, run the following script.
    ```bash
    cd /path/to/dhd/hicodet
    bash download.sh
    ```
    2. If you have previously downloaded the dataset, simply create a soft link.
    ```bash
    cd /path/to/dhd/hicodet
    ln -s /path/to/hicodet_20160224_det ./hico_20160224_det
    ```
4. Prepare the V-COCO dataset (contained in [MS COCO](https://cocodataset.org/#download)).
    1. If you have not downloaded the dataset before, run the following script
    ```bash
    cd /path/to/dhd/vcoco
    bash download.sh
    ```
    2. If you have previously downloaded the dataset, simply create a soft link
    ```bash
    cd /path/to/dhd/vcoco
    ln -s /path/to/coco ./mscoco2014
    ```
5. Prepare the VG dataset from [VG](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html).
    1. If you have downloaded the dataset, simply create a soft link
    ```bash
    cd /path/to/dhd/vg
    ln -s /path/to/vg ./vg
    ```
6. Prepare the preprocessed annotations, from [vg](https://drive.google.com/drive/folders/14SjjGDNTKg5KAGAuqNGSJQrdm7a84krV?usp=sharing) and [hicodet](https://drive.google.com/drive/folders/1U76Vj7sPKjmINly-OeHS4kZRDXE4ftRJ?usp=sharing), and put them into the corresponding dataset directory.

## License
DHD is released under the [BSD-3-Clause License](./LICENSE).


## Training and Testing

Refer to [`train.sh`](./train.sh) for training and [`test.sh`](./test.sh) for testing commands with different options.  

## Citation
```
@inproceedings{wu2024toward,
  title={Toward Open-Set Human Object Interaction Detection},
  author={Wu, Mingrui and Liu, Yuqi and Ji, Jiayi and Sun, Xiaoshuai and Ji, Rongrong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={6066--6073},
  year={2024}
}
```

## Acknowledge
This repo is based on [UPT](https://github.com/fredzzhang/upt).


