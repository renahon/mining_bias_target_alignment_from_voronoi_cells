# Mining bias-target Alignment from Voronoi Cells

[![paper](https://img.shields.io/badge/ICCV-paper-blue)](https://openaccess.thecvf.com/content/ICCV2023/papers/Nahon_Mining_bias-target_Alignment_from_Voronoi_Cells_ICCV_2023_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2305.03691-b31b1b.svg)](https://arxiv.org/abs/2305.03691)

The official repository. Please cite as
```
@InProceedings{Nahon_2023_ICCV,
    author    = {Nahon, R\'emi and Nguyen, Van-Tam and Tartaglione, Enzo},
    title     = {Mining bias-target Alignment from Voronoi Cells},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {4946-4955}
}
```

This is an example implementation of : Mining bias-target Alignment from Voronoi Cells.

This repository is tailored for Biased-MNIST but can be adapted to other datasets.

## Installations

If you want to work with conda :

```bash
conda create -n voronoi_cells_bias_alignment python=3.9
conda activate voronoi_cells_bias_alignment
pip3 install -r pip_requirements.txt
```

If not, simply install the requirement with :

```bash
pip3 install -r pip_requirements.txt
```

## Run our method on Biased-MNIST

1. In a terminal, get inside the voronoi_cells_bias_alignment folder

2. Run : python3 main.py

### Possible extra arguments #####

- --dev : to specify the device you want to work on (ex: cpu or cuda:0)

- --rho : to select a level of digit-color correlation in Biased-MNIST (ex : 0.997)

- other extra arguments detailed in utils/configs.py
